from itertools import groupby

import numpy as np


def isin(event_a, event_b):
    """判断event_a是否event_b的一个子集"""
    if event_a["event_type"] != event_b["event_type"]:
        return False
    for argu in event_a["arguments"]:
        if argu not in event_b["arguments"]:
            return False
    return True


class DedupList(list):
    """定义去重的list"""

    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """构建邻集（host节点与其所有邻居的集合）"""
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def postprocess_gplinker(
    args, batch_outputs, offset_mappings, texts, trigger=True, threshold=0
):
    batch_results = []
    for argu_output, head_output, tail_output, offset_mapping, text in zip(
        batch_outputs[0].cpu().numpy(),
        batch_outputs[1].cpu().numpy(),
        batch_outputs[2].cpu().numpy(),
        offset_mappings,
        texts,
    ):
        argus = set()
        argu_output[:, [0, -1]] -= np.inf
        argu_output[:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(argu_output > threshold)):
            argus.add(args.labels[l] + (h, t))

        # 构建链接
        links = set()
        for i1, (_, _, h1, t1) in enumerate(argus):
            for i2, (_, _, h2, t2) in enumerate(argus):
                if i2 > i1:
                    if head_output[0, min(h1, h2), max(h1, h2)] > threshold:
                        if tail_output[0, min(t1, t2), max(t1, t2)] > threshold:
                            links.add((h1, t1, h2, t2))
                            links.add((h2, t2, h1, t1))

        # 析出事件
        events = []
        for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
            for event in clique_search(list(sub_argus), links):
                events.append([])
                for argu in event:
                    start, end = (
                        offset_mapping[argu[2]][0],
                        offset_mapping[argu[3]][1],
                    )
                    events[-1].append(
                        [argu[0], argu[1], text[start:end], f"{start};{end}"]
                    )
                if trigger and all([argu[1] != "触发词" for argu in event]):
                    events.pop()

        batch_results.append(events)

    return batch_results

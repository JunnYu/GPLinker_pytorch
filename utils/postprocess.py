import numpy as np
import torch


def postprocess_gplinker(args, batch_outputs, offset_mappings, texts, threshold=0):
    batch_results = []
    for entity_output, head_output, tail_output, offset_mapping, text in zip(
        batch_outputs[0].cpu().numpy(),
        batch_outputs[1].cpu().numpy(),
        batch_outputs[2].cpu().numpy(),
        offset_mappings,
        texts,
    ):
        subjects, objects = set(), set()
        entity_output[:, [0, -1]] -= np.inf
        entity_output[:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(entity_output > threshold)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))

        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(head_output[:, sh, oh] > threshold)[0]
                p2s = np.where(tail_output[:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    try:
                        triplet = (
                            text[offset_mapping[sh][0] : offset_mapping[st][1]],
                            args.id2predicate[p],
                            text[offset_mapping[oh][0] : offset_mapping[ot][1]],
                        )
                        spoes.add(triplet)
                    except Exception as e:
                        print(e)
        batch_results.append(list(spoes))
    return batch_results


def get_spots_fr_shaking_tag(shaking_idx2matrix_idx, shaking_outputs):
    """
    shaking_tag -> spots
    shaking_tag: (shaking_seq_len, tag_id)
    spots: [(start_ind, end_ind, tag_id), ]
    """
    spots = []
    pred_shaking_tag = (shaking_outputs > 0.0).long()
    nonzero_points = torch.nonzero(pred_shaking_tag, as_tuple=False)
    for point in nonzero_points:
        shaking_idx, tag_idx = point[0].item(), point[1].item()
        pos1, pos2 = shaking_idx2matrix_idx[shaking_idx]
        spot = (pos1, pos2, tag_idx)
        spots.append(spot)
    return spots


def postprocess_tplinker_plus(args, batch_outputs, offset_mappings, texts, seqlen):
    batch_results = []
    for shaking_outputs, offset_mapping, text in zip(
        batch_outputs, offset_mappings, texts
    ):
        shaking_idx2matrix_idx = [
            (ind, end_ind)
            for ind in range(seqlen)
            for end_ind in list(range(seqlen))[ind:]
        ]
        head_ind2entities = {}
        rel_list = []

        matrix_spots = get_spots_fr_shaking_tag(
            shaking_idx2matrix_idx, shaking_outputs.cpu()
        )
        for sp in matrix_spots:
            tag = args.id2tag[sp[2]]
            ent_type, link_type = tag.split("=")
            # for an entity, the start position can not be larger than the end pos.
            if link_type != "EH2ET" or sp[0] > sp[1]:
                continue

            entity = {
                "type": ent_type,
                "tok_span": [sp[0], sp[1]],
            }
            # take ent_head_pos as the key to entity list
            head_key = sp[0]
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)

        # tail link
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = args.id2tag[sp[2]]
            rel, link_type = tag.split("=")

            if link_type == "ST2OT":
                rel = args.predicate2id[rel]
                tail_link_memory = (rel, sp[0], sp[1])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                rel = args.predicate2id[rel]
                tail_link_memory = (rel, sp[1], sp[0])
                tail_link_memory_set.add(tail_link_memory)

        # head link
        for sp in matrix_spots:
            tag = args.id2tag[sp[2]]
            rel, link_type = tag.split("=")

            if link_type == "SH2OH":
                rel = args.predicate2id[rel]
                subj_head_key, obj_head_key = sp[0], sp[1]
            elif link_type == "OH2SH":
                rel = args.predicate2id[rel]
                subj_head_key, obj_head_key = sp[1], sp[0]
            else:
                continue

            if (
                subj_head_key not in head_ind2entities
                or obj_head_key not in head_ind2entities
            ):
                # no entity start with subj_head_key and obj_head_key
                continue

            # all entities start with this subject head
            subj_list = head_ind2entities[subj_head_key]
            # all entities start with this object head
            obj_list = head_ind2entities[obj_head_key]

            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = (rel, subj["tok_span"][1], obj["tok_span"][1])

                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue

                    rel_list.append(
                        (
                            text[
                                offset_mapping[subj["tok_span"][0]][0] : offset_mapping[
                                    subj["tok_span"][1]
                                ][1]
                            ],
                            args.id2predicate[rel],
                            text[
                                offset_mapping[obj["tok_span"][0]][0] : offset_mapping[
                                    obj["tok_span"][1]
                                ][1]
                            ],
                        )
                    )

        batch_results.append(rel_list)
    return batch_results

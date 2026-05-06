from itertools import combinations


def superset_filtering(rules):
    ordered = sorted(
        rules,
        key=lambda r: (-r.conf, -r.sup, len(r.itemset), r.ans, r.itemset),
    )

    kept = []
    seen_itemsets = set()
    kept_itemsets_by_answer = {}

    for rule in ordered:
        itemset = tuple(sorted(rule.itemset))
        if itemset in seen_itemsets:
            continue

        kept_itemsets = kept_itemsets_by_answer.setdefault(rule.ans, set())
        redundant = False
        for subset_size in range(len(itemset)):
            for subset in combinations(itemset, subset_size):
                if subset in kept_itemsets:
                    redundant = True
                    break
            if redundant:
                break

        if redundant:
            continue

        kept.append(rule)
        kept_itemsets.add(itemset)
        seen_itemsets.add(itemset)

    return kept

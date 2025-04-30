def get_family_input():
    family = []
    while True:
        relation = input("Relation (or 'done' to finish): ").strip().lower()
        if relation == 'done':
            break
        if relation not in ['mother', 'father', 'sibling', 'grandparent', 'aunt_uncle', 'cousin', 'child']:
            continue
        diabetes_input = input(f"Does your {relation} have diabetes? (yes/no): ").strip().lower()
        if diabetes_input not in ['yes', 'no']:
            continue
        has_diabetes = diabetes_input == 'yes'
        family.append((relation, has_diabetes))
    return family

def calculate_approx_dpf(diabetic_relatives):
    weights = {
        'mother': 0.5,
        'father': 0.5,
        'sibling': 0.5,
        'grandparent': 0.25,
        'aunt_uncle': 0.25,
        'cousin': 0.125,
        'child': 0.5
    }

    total_weight = 0
    diabetic_weight = 0

    for relation, has_diabetes in diabetic_relatives:
        weight = weights.get(relation, 0)
        total_weight += weight
        if has_diabetes:
            diabetic_weight += weight

    if total_weight == 0:
        return 0.0
    return round(diabetic_weight / total_weight, 3)

if __name__ == "__main__":
    family_history = get_family_input()
    dpf_score = calculate_approx_dpf(family_history)
    print(dpf_score)

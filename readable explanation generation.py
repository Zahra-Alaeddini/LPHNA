import pandas as pd
import numpy as np

def generate_human_readable_explanation(row):
    src, tgt = row["Predicted_Link"].split("->")
    src_name = src.split("::")[1] if "::" in src else src
    tgt_name = tgt.split("::")[1] if "::" in tgt else tgt
    metapath = row["Metapath"]
    narrative = row["Narrative"]
    cf_narrative = row["Counterfactual_Narrative"]
    cf_delta = row["Counterfactual_Delta"]
    contrastive_narratives = row["Contrastive_Narratives"].split(" || ") if row["Contrastive_Narratives"] else []
    final_score = row["Final_Score"]

    relation_types = {
        "CtD": "treats the disease",
        "CpD": "palliates the disease",
        "CrC": "is similar to another compound",
        "DrD": "is similar to another disease"
    }
    rel_name = "is related to"
    for rel_type, description in relation_types.items():
        if rel_type in metapath:
            rel_name = description
            break

    explanation = f"Why {src_name} is predicted to {rel_name} {tgt_name}:\n"

    metapath_desc = {
        "CtD": "a compound treating a related disease",
        "CpD": "a compound palliating a related disease",
        "CrC": "similar compounds with shared properties",
        "DrD": "diseases with similar characteristics"
    }
    metapath_context = next((desc for rel, desc in metapath_desc.items() if rel in metapath), "a specific pattern")
    narrative_simplified = narrative.replace(f"Link {src}->{tgt} via {metapath}",
                                            f"The model identified a strong connection through {metapath_context} ({metapath})")
    narrative_simplified = narrative_simplified.replace("(attn:", "with high importance (")
    narrative_simplified = narrative_simplified.replace(", contrib:", ", contributing")
    narrative_simplified = narrative_simplified.replace(").", f") to a confidence of {final_score:.2f}.")
    explanation += f"- {narrative_simplified}\n"

    normalized_delta = min(abs(cf_delta), 1.0) if abs(cf_delta) > 1 else abs(cf_delta)
    cf_simplified = cf_narrative.replace(
        f"Removing path {row['Explanation_Path']} reduces common neighbors by {cf_delta:.1f}, changing score by {abs(cf_delta):.3f}",
        f"If the connection through {metapath_context} ({metapath}) were removed, the link would weaken, reducing confidence by about {normalized_delta:.2f}")
    explanation += f"- {cf_simplified}.\n"

    contrastive_simplified = ""
    if contrastive_narratives:
        best_contrast = min(
            contrastive_narratives,
            key=lambda x: abs(float(x.split("delta_deg=")[1].split(",")[0])),
            default=contrastive_narratives[0]
        )
        contrast_tgt = best_contrast.split(" vs ")[1].split(":")[0]
        contrast_tgt_name = contrast_tgt.split("::")[1] if "::" in contrast_tgt else contrast_tgt
        shared_count = best_contrast.count("->")
        contrastive_simplified = (
            f"- Compared to {contrast_tgt_name}, {tgt_name} has {shared_count} similar connection patterns "
            f"with {src_name}, but is favored due to stronger or more relevant connections for {rel_name}.\n"
        )
    explanation += contrastive_simplified

    explanation += (
        f"In summary, the model predicts that {src_name} {rel_name} {tgt_name} with a confidence of {final_score:.2f}, "
        f"driven by the strong connection through {metapath_context} ({metapath}), which stands out compared to alternatives.\n"
    )

    return explanation

input_csv = "causal_heteroxplain_explanations_hebbian.csv"
df = pd.read_csv(input_csv)

results = []
for idx, row in df.iterrows():
    explanation = generate_human_readable_explanation(row)
    results.append({
        "Predicted_Link": row["Predicted_Link"],
        "Relation": row["Metapath"],
        "Explanation": explanation,
        "Confidence_Score": row["Final_Score"]
    })

output_csv = "readable_explanations.csv"
explanation_df = pd.DataFrame(results)
explanation_df.to_csv(output_csv, index=False)
print(f"readable explanations saved to {output_csv}")
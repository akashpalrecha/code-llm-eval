import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def load_data(root_dir: str) -> pd.DataFrame:
    data = []
    root_path = Path(root_dir)
    for model_path in root_path.iterdir():
        if model_path.is_dir():
            for json_file in model_path.glob("*-eval.json"):
                with json_file.open("r") as f:
                    content = json.load(f)
                    benchmark = json_file.stem.replace("-eval", "")
                    metrics = content.get(benchmark, {})
                    entry = {"Model": model_path.name, "Benchmark": benchmark}
                    for k, v in metrics.items():
                        entry[k] = v
                    data.append(entry)
    return pd.DataFrame(data)


def main():
    st.title("LLM Evaluation Dashboard")

    st.sidebar.header("Configuration")
    root_dir = st.sidebar.text_input("Directory containing evaluation results:", value="sweep-run/version_0")

    if not Path(root_dir).exists():
        st.error(f"The directory '{root_dir}' does not exist.")
        return

    df = load_data(root_dir)

    if df.empty:
        st.warning("No data found. Please check your directory structure and JSON files.")
        return

    # Selection widgets
    models = df["Model"].unique().tolist()
    benchmarks = df["Benchmark"].unique().tolist()
    k_values = [col for col in df.columns if col.startswith("pass@")]

    selected_models = st.sidebar.multiselect("Select Models:", models, default=models)
    selected_benchmarks = st.sidebar.multiselect("Select Benchmarks:", benchmarks, default=benchmarks)
    selected_k_values = st.sidebar.multiselect("Select Pass@k Metrics:", k_values, default=k_values)

    # Filter the DataFrame based on selections
    filtered_df = df[(df["Model"].isin(selected_models)) & (df["Benchmark"].isin(selected_benchmarks))]

    if filtered_df.empty:
        st.warning("No data matches the selected criteria.")
        return

    # Prepare data for plotting
    plot_df = filtered_df.melt(
        id_vars=["Benchmark", "Model"], value_vars=selected_k_values, var_name="Pass@k", value_name="Metric Value"
    )

    # Ensure Pass@k is ordered numerically
    plot_df["Pass@k"] = plot_df["Pass@k"].astype("category")
    plot_df["Pass@k"] = plot_df["Pass@k"].cat.set_categories(selected_k_values, ordered=True)

    # Define unique markers and colors for models
    shapes = [
        "circle",
        "square",
        "triangle-up",
        "diamond",
        "cross",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "hexagon",
    ]
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
    ]
    model_shapes = {model: shapes[i % len(shapes)] for i, model in enumerate(models)}
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    for benchmark in selected_benchmarks:
        st.write(f"### {benchmark}")
        benchmark_df = plot_df[plot_df["Benchmark"] == benchmark]

        # Create an Altair scatter plot
        chart = (
            alt.Chart(benchmark_df)
            .mark_point(filled=True, size=100)
            .encode(
                x=alt.X("Pass@k", title="Pass@k"),
                y=alt.Y("Metric Value", title="Metric Value"),
                shape=alt.Shape(
                    "Model",
                    scale=alt.Scale(domain=models, range=[model_shapes[m] for m in models]),
                    legend=alt.Legend(title="Model"),
                ),
                color=alt.Color(
                    "Model",
                    scale=alt.Scale(domain=models, range=[model_colors[m] for m in models]),
                    legend=None,
                ),
                tooltip=["Model", "Pass@k", "Metric Value"],
            )
            .properties(width=600, height=400)
            .interactive()
        )

        st.altair_chart(chart)


if __name__ == "__main__":
    main()

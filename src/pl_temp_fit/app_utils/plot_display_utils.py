import emcee
import streamlit as st

from pl_temp_fit import config_utils, plot_utils
from pl_temp_fit.app_utils.plot_utils_interactive import (
    plot_state_diagram_interactive,
    plot_fit_statistics_interactive,
    plot_chains_interactive,
    plot_distribution_interactive,
    plot_lifetime_interactive,
    plot_blobs_chain_interactive,
    plot_corner_interactive,
)


def read_results(test_id, databse_path):
    model_config, model_config_save = config_utils.load_model_config(
        test_id, database_folder=databse_path
    )

    filename = model_config_save["save_folder"] + "/sampler.h5"
    reader = emcee.backends.HDFBackend(filename, name="multi_core")

    return reader, model_config, model_config_save


def display_data_info():
    """Display basic information about loaded data."""
    reader = st.session_state.get("reader")
    model_config_save = st.session_state.get("model_config_save")

    if reader and model_config_save:
        with st.expander("📊 Data Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Samples", reader.get_chain().shape[0])
                st.metric("Dimensions", reader.get_chain().shape[2])
            with col2:
                st.metric("Chains", reader.get_chain().shape[1])
                blobs_info = (
                    reader.get_blobs().dtype.names
                    if reader.get_blobs() is not None
                    else "None"
                )
                if blobs_info != "None":
                    st.write(f"**Blobs:** {len(blobs_info)} fields")
                    with st.expander("Blob fields"):
                        st.write(list(blobs_info))
                else:
                    st.write("**Blobs:** None")


def get_common_controls():
    """Get common controls used across multiple plots."""
    controls = {}
    controls["discard"] = st.slider(
        "🔧 Discard initial samples",
        min_value=0,
        max_value=1000,
        value=50,
        help="Number of initial samples to discard (burn-in)",
    )
    controls["filter_log_likelihood"] = st.text_input(
        "🎯 Filter condition",
        placeholder="e.g., blobs['log_likelihood_spectra'] > -100",
        help="Python expression to filter samples",
    )
    return controls


def plot_selection_tabbed():
    """Streamlit interface with tabbed layout for better organization."""

    # Check if data is loaded
    reader = st.session_state.get("reader")
    model_config_save = st.session_state.get("model_config_save")
    model_config = st.session_state.get("model_config")

    if not reader:
        st.warning("⚠️ Please load data first to access plotting functions.")
        return

    # Display data information
    display_data_info()

    # Create tabs for different plot categories
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Fit Analysis",
            "🔗 Chain Analysis",
            "📊 Distributions",
            "🔬 Advanced",
        ]
    )

    with tab1:
        st.header("Fit Statistics & Experimental Data")

        # Create columns for better layout
        settings_col, plot_col = st.columns([1, 2])

        with settings_col:
            st.subheader("⚙️ Settings")
            plot_type = st.selectbox(
                "Plot Type",
                [
                    "Fit Statistics",
                    "Fit to Experimental Data",
                    "Fit Statistics Multi",
                ],
            )

            if plot_type == "Fit Statistics":
                discard = st.slider("Discard", 0, 1000, 0)
                range_log_prior = st.slider(
                    "Log Prior Range", -1000, 0, (-500, 0)
                )
                range_chi_square = st.slider(
                    "Chi Square Range", 0, 20, (0, 10)
                )
                filter_log_likelihood = st.checkbox(
                    "Filter Log Likelihood", value=True
                )

                if st.button("Generate Plot", type="primary"):
                    with plot_col, st.spinner("Generating plot..."):
                        fig = plot_fit_statistics_interactive(
                            reader,
                            discard=discard,
                            range_log_prior=range_log_prior,
                            range_chi_square=range_chi_square,
                            filter_log_likelihood=filter_log_likelihood,
                        )
                        st.plotly_chart(fig, width='stretch')

            elif plot_type == "Fit to Experimental Data":
                discard = st.slider("Discard", 0, 1000, 300)
                filter_condition = st.text_input(
                    "Filter Condition",
                    "blobs['log_likelihood_spectra'] > max(blobs['log_likelihood_spectra']) * 1.1",
                )

                if (
                    st.button("Generate Plot", type="primary")
                    and model_config_save
                    and model_config
                ):
                    with plot_col:
                        with st.spinner("Generating plot..."):
                            fig, _ = plot_utils.plot_fit_to_experimental_data(
                                model_config_save,
                                model_config,
                                reader,
                                discard=discard,
                                filter_log_likelihood=filter_condition or (),
                            )
                            st.pyplot(fig)

    with tab2:
        st.header("Chain Convergence Analysis")

        settings_col, plot_col = st.columns([1, 2])

        with settings_col:
            st.subheader("⚙️ Settings")
            chain_plot_type = st.selectbox(
                "Chain Plot Type",
                ["All Chains", "Selected Chains", "Blobs & Lifetime"],
            )

            if chain_plot_type == "All Chains":
                discard = st.slider(
                    "Discard", 0, 1000, 0, key="chains_discard"
                )

                if (
                    st.button("Generate Chain Plot", type="primary")
                    and model_config_save
                ):
                    with plot_col, st.spinner("Generating plot..."):
                        fig = plot_chains_interactive(reader, model_config_save, discard=discard)
                        st.plotly_chart(fig, width='stretch')

            elif chain_plot_type == "Selected Chains":
                discard = st.slider(
                    "Discard", 0, 1000, 50, key="diff_chains_discard"
                )
                chains_input = st.text_input(
                    "Chain IDs (comma separated)", "2,15,3,4"
                )
                chains_list = [
                    int(x.strip())
                    for x in chains_input.split(",")
                    if x.strip().isdigit()
                ]

                if (
                    st.button("Generate Selected Chains Plot", type="primary")
                    and model_config_save
                ):
                    with plot_col:
                        with st.spinner("Generating plot..."):
                            fig, _ = plot_utils.plot_diff_chains(
                                reader,
                                model_config_save,
                                discard=discard,
                                chains_list=chains_list or [2, 15, 3, 4],
                            )
                            st.pyplot(fig)

            elif chain_plot_type == "Blobs & Lifetime":
                discard = st.slider(
                    "Discard", 0, 1000, 100, key="blobs_discard"
                )
                temp_lifetimes = st.session_state.get(
                    "temperature_lifetimes_exp", [60, 140, 220, 300]
                )
                st.write(f"Temperature lifetimes: {temp_lifetimes}")

                if st.button("Generate Blobs Plot", type="primary"):
                    with plot_col, st.spinner("Generating plot..."):
                        fig = plot_blobs_chain_interactive(
                            reader,
                            discard=discard,
                            temp_lifetime=temp_lifetimes,
                        )
                        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.header("Parameter Distributions")

        settings_col, plot_col = st.columns([1, 2])

        with settings_col:
            st.subheader("⚙️ Settings")
            dist_plot_type = st.selectbox(
                "Distribution Plot Type",
                [
                    "Parameter Distribution",
                    "Corner Plot",
                    "Lifetime Distribution",
                ],
            )

            if dist_plot_type == "Parameter Distribution":
                discard = st.slider("Discard", 0, 1000, 50, key="dist_discard")
                filter_condition = st.text_input(
                    "Filter Condition", "", key="dist_filter"
                )

                if (
                    st.button("Generate Distribution Plot", type="primary")
                    and model_config_save
                ):
                    with plot_col, st.spinner("Generating plot..."):
                        fig = plot_distribution_interactive(
                            reader,
                            model_config_save,
                            discard=discard,
                            filter_log_likelihood=filter_condition or "",
                        )
                        st.plotly_chart(fig, width='stretch')

            elif dist_plot_type == "Corner Plot":
                discard = st.slider(
                    "Discard", 0, 1000, 0, key="corner_discard"
                )
                filter_condition = st.text_input(
                    "Filter Condition", "", key="corner_filter"
                )

                if (
                    st.button("Generate Corner Plot", type="primary")
                    and model_config_save
                ):
                    with plot_col, st.spinner("Generating plot..."):
                        fig = plot_corner_interactive(
                            reader,
                            model_config_save,
                            discard=discard,
                            filter_log_likelihood=filter_condition or "",
                        )
                        st.plotly_chart(fig, width='stretch')

            elif dist_plot_type == "Lifetime Distribution":
                discard = st.slider(
                    "Discard", 0, 1000, 100, key="lifetime_discard"
                )
                temperature = st.number_input(
                    "Temperature (K)",
                    value=300.0,
                    min_value=0.0,
                    format="%0.1f",
                    key="lifetime_temp",
                    step=0.1,
                )
                filter_log_likelihood = st.checkbox(
                    "Filter Log Likelihood", key="lifetime_filter"
                )

                if st.button("Generate Lifetime Plot", type="primary"):
                    with plot_col, st.spinner("Generating plot..."):
                        fig = plot_lifetime_interactive(
                            reader,
                            discard=discard,
                            temperature=temperature,
                            filter_log_likelihood=filter_log_likelihood,
                        )
                        st.plotly_chart(fig, width='stretch')

    with tab4:
        st.header("Advanced Analysis")
        st.info("💡 Advanced plotting options for multi-dataset comparison")

        settings_col, plot_col = st.columns([1, 2])

        with settings_col:
            st.subheader("⚙️ Analysis Type")
            analysis_type = st.selectbox(
                "Analysis Type",
                [
                    "Model Parameters", 
                    "Multi-Dataset Comparison"
                ]
            )
            
            if analysis_type == "Model Parameters":
                st.subheader("⚡ State Diagram")
                st.info("Visualize electronic state energy levels and transitions")
                
                if st.button("Generate State Diagram", type="primary"):
                    if model_config_save:
                        with plot_col, st.spinner("Generating state diagram..."):
                            fig = plot_state_diagram_interactive(model_config_save)
                            st.plotly_chart(fig, width='stretch')
                    else:
                        st.error("❌ Model configuration not available")
            
            elif analysis_type == "Multi-Dataset Comparison":
                st.subheader("⚙️ Multi-Plot Settings")
            multi_plot_type = st.selectbox(
                "Multi-Plot Type",
                ["Distribution Multi", "Fit Statistics Multi"],
            )

            if multi_plot_type == "Distribution Multi":
                discard = st.slider(
                    "Discard", 0, 1000, 200, key="multi_dist_discard"
                )
                color = st.color_picker("Plot Color", "#1f77b4")
                legend_label = st.text_input("Legend Label", "Dataset 1")

                if (
                    st.button(
                        "Generate Multi Distribution Plot", type="primary"
                    )
                    and model_config_save
                ):
                    with plot_col:
                        with st.spinner("Generating plot..."):
                            fig, _ = plot_utils.plot_distribution_multi(
                                reader,
                                model_config_save,
                                discard=discard,
                                filter_log_likelihood=True,
                                color=color,
                                legend_label=legend_label,
                            )
                            st.pyplot(fig)

            elif multi_plot_type == "Fit Statistics Multi":
                discard = st.slider(
                    "Discard", 0, 1000, 200, key="multi_fit_discard"
                )
                range_log_prior = st.slider(
                    "Log Prior Range",
                    -1000,
                    0,
                    (-500, 0),
                    key="multi_log_prior",
                )
                range_chi_square = st.slider(
                    "Chi Square Range", 0, 20, (0, 3), key="multi_chi_square"
                )
                color = st.color_picker(
                    "Plot Color", "#ff7f0e", key="multi_color"
                )
                legend_label = st.text_input(
                    "Legend Label", "Dataset 1", key="multi_legend"
                )

                if st.button(
                    "Generate Multi Fit Statistics Plot", type="primary"
                ):
                    with plot_col:
                        with st.spinner("Generating plot..."):
                            fig, _ = plot_utils.plot_fit_statistics_multi(
                                reader,
                                discard=discard,
                                range_log_prior=range_log_prior,
                                range_chi_square=range_chi_square,
                                filter_log_likelihood=True,
                                legend_label=legend_label,
                                color=color,
                            )
                            st.pyplot(fig)


def plot_selection():
    """Legacy function maintained for backward compatibility."""
    plot_selection_tabbed()

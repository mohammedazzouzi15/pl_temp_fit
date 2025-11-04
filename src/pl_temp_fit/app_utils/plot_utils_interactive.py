"""Interactive plotting utilities using Plotly for enhanced user experience."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_experimental_data_interactive(
    exp_data, temperature_list, hws, title="Experimental Data"
):
    """Create interactive plot of experimental PL data."""
    # Convert temperature list to numeric for sorting - handle both string and numeric formats
    temp_values = []
    for temp in temperature_list:
        if isinstance(temp, str):
            # If it's a string, remove 'K' and convert to float
            temp_values.append(float(temp.replace("K", "")))
        else:
            # If it's already numeric, use as is
            temp_values.append(float(temp))

    temp_sorted_indices = np.argsort(temp_values)

    # Create subplot
    fig = go.Figure()

    # Color palette for different temperatures
    colors = px.colors.qualitative.Set1

    # Plot each temperature
    for idx, temp_idx in enumerate(temp_sorted_indices):
        temp = temperature_list[temp_idx]
        temp_value = temp_values[temp_idx]

        # Get the data for this temperature
        y_data = exp_data[:, temp_idx]

        fig.add_trace(
            go.Scatter(
                x=hws,
                y=y_data,
                mode="lines+markers",
                name=f"{temp_value:.0f} K",
                line={"color": colors[idx % len(colors)], "width": 2},
                marker={"size": 4, "color": colors[idx % len(colors)]},
                hovertemplate=(
                    f"<b>{temp_value:.0f} K</b><br>"
                    "Energy: %{x:.3f} eV<br>"
                    "Intensity: %{y:.3e}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title=f"📊 {title}",
        title_x=0.5,
        xaxis_title="Energy (eV)",
        yaxis_title="PL Intensity (a.u.)",
        hovermode="closest",
        height=500,
        showlegend=True,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Style axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        # type="log"  # Log scale for PL intensity
    )

    return fig


def plot_data_comparison_interactive(
    original_data,
    modified_data,
    original_hws,
    modified_hws,
    temperature_list,
    original_title="Original",
    modified_title="Modified",
):
    """Create interactive comparison plot of original vs modified data."""
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[original_title, modified_title],
        shared_yaxes=True,
        horizontal_spacing=0.1,
    )

    # Color palette
    colors = px.colors.qualitative.Set1

    # Convert temperature list to numeric for sorting - handle both string and numeric formats
    temp_values = []
    for temp in temperature_list:
        if isinstance(temp, str):
            # If it's a string, remove 'K' and convert to float
            temp_values.append(float(temp.replace("K", "")))
        else:
            # If it's already numeric, use as is
            temp_values.append(float(temp))

    temp_sorted_indices = np.argsort(temp_values)

    # Plot original data
    for idx, temp_idx in enumerate(temp_sorted_indices):
        temp = temperature_list[temp_idx]
        temp_value = temp_values[temp_idx]
        color = colors[idx % len(colors)]

        # Original data
        fig.add_trace(
            go.Scatter(
                x=original_hws,
                y=original_data[:, temp_idx],
                mode="lines",
                name=f"{temp_value:.0f} K",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{temp_value:.0f} K (Original)</b><br>"
                    "Energy: %{x:.3f} eV<br>"
                    "Intensity: %{y:.3e}<br>"
                    "<extra></extra>"
                ),
                legendgroup=f"temp_{temp_value}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Modified data
        fig.add_trace(
            go.Scatter(
                x=modified_hws,
                y=modified_data[:, temp_idx],
                mode="lines",
                name=f"{temp_value:.0f} K",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{temp_value:.0f} K (Modified)</b><br>"
                    "Energy: %{x:.3f} eV<br>"
                    "Intensity: %{y:.3e}<br>"
                    "<extra></extra>"
                ),
                legendgroup=f"temp_{temp_value}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title="🔄 Data Comparison: Original vs Modified",
        title_x=0.5,
        height=500,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Update axes
    fig.update_xaxes(
        title_text="Energy (eV)",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        title_text="PL Intensity (a.u.)",
        # type="log",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig


def plot_fit_statistics_interactive(
    reader,
    range_chi_square=(0, 3),
    range_log_prior=(-1000, 0),
    discard=5,
    filter_log_likelihood="",
):
    """Interactive plot of fit statistics from sampling output."""
    print("number of iterations", reader.iteration)
    blobs = reader.get_blobs(flat=True, discard=discard)
    distribution = reader.get_chain(discard=discard, flat=True)
    print(max(blobs["log_likelihood_spectra"]))

    if filter_log_likelihood:
        distribution = eval(f" distribution[{filter_log_likelihood}]")

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Chi Square Distribution",
            "Log Likelihood Distribution",
            "Log Likelihood vs Iteration",
        ],
        specs=[
            [
                {"secondary_y": False},
                {"secondary_y": False},
                {"secondary_y": False},
            ]
        ],
    )

    # Chi square histogram
    fig.add_trace(
        go.Histogram(
            x=blobs["Chi square_spectra"],
            nbinsx=30,
            name="Chi Square",
            marker_color="blue",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Log likelihood histogram
    fig.add_trace(
        go.Histogram(
            x=blobs["log_likelihood_spectra"],
            nbinsx=30,
            name="Log Likelihood",
            marker_color="green",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Log likelihood vs iteration
    fig.add_trace(
        go.Scatter(
            x=list(range(len(blobs["log_likelihood_spectra"]))),
            y=blobs["log_likelihood_spectra"],
            mode="lines",
            name="Log Likelihood",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # Update layout
    fig.update_xaxes(
        title_text="Chi Square Spectra", row=1, col=1, range=range_chi_square
    )
    fig.update_xaxes(
        title_text="Log Likelihood Spectra",
        row=1,
        col=2,
        range=range_log_prior,
    )
    fig.update_xaxes(title_text="Iteration", row=1, col=3)

    fig.update_yaxes(title_text="Number of Samples", row=1, col=1)
    fig.update_yaxes(title_text="Number of Samples", row=1, col=2)
    fig.update_yaxes(title_text="Log Likelihood Spectra", row=1, col=3)

    fig.update_layout(
        height=400,
        title_text="📊 Fit Statistics Analysis",
        title_x=0.5,
        showlegend=False,
    )

    return fig


def plot_chains_interactive(reader, model_config_save, discard=50):
    """Interactive plot of MCMC chains."""
    csv_name = model_config_save["csv_name_pl"]
    label_list = []
    for key in model_config_save["params_to_fit_init"].keys():
        label_list.extend(
            [
                key + "_" + x
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )

    params_to_fit = model_config_save["params_to_fit_init"]
    min_bound = {}
    max_bound = {}
    for key in params_to_fit.keys():
        for keys in params_to_fit[key].keys():
            min_bound[f"{key}_{keys}"] = model_config_save["min_bounds"][key][
                keys
            ]
            max_bound[f"{key}_{keys}"] = model_config_save["max_bounds"][key][
                keys
            ]

    labels = label_list
    samples = reader.get_chain(discard=discard)

    # Create subplots
    fig = make_subplots(
        rows=len(labels),
        cols=1,
        subplot_titles=labels,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    colors = px.colors.qualitative.Set3
    colors = [c for c in px.colors.qualitative.Set3 if c.lower() not in ['#ffffb3', '#ffffb3'.lower(), 'rgb(255, 255, 179)', 'rgb(255,255,179)']]

    for i, label in enumerate(labels):
        # Plot all chains for this parameter
        for chain_idx in range(samples.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(samples[:, chain_idx, i]))),
                    y=samples[:, chain_idx, i],
                    mode="lines",
                    name=f"Chain {chain_idx}" if i == 0 else "",
                    line=dict(color=colors[chain_idx % len(colors)], width=1),
                    opacity=0.7,
                    showlegend=False,  # Only show legend for first parameter
                    hovertemplate=f"{label}<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>,<br>chain={chain_idx}",
                ),
                row=i + 1,
                col=1,
            )

        # Set y-axis range if bounds are available
        if min_bound[label] is not None and max_bound[label] is not None:
            fig.update_yaxes(
                range=[min_bound[label], max_bound[label]], row=i + 1, col=1
            )

        fig.update_yaxes(title_text=label, row=i + 1, col=1)

    fig.update_xaxes(title_text="Step Number", row=len(labels), col=1)
    fig.update_layout(
        height=150 * len(labels),
        title_text=f"🔗 MCMC Chains - {csv_name.split('/')[-1]}",
        title_x=0.5,
    )

    return fig


def plot_distribution_interactive(
    reader,
    model_config_save,
    discard=10,
    filter_log_likelihood="",
):
    """Interactive plot of parameter distributions."""
    csv_name = model_config_save["csv_name_pl"]
    min_bounds_list, max_bounds_list = [], []
    label_list = []

    for key in model_config_save["params_to_fit_init"].keys():
        label_list.extend(
            [
                key + "_" + x
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )
        min_bounds_list.extend(
            [
                model_config_save["min_bounds"][key][x]
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )
        max_bounds_list.extend(
            [
                model_config_save["max_bounds"][key][x]
                for x in model_config_save["params_to_fit_init"][key].keys()
            ]
        )

    labels = label_list
    ndim = len(labels)

    distribution = reader.get_chain(discard=discard, flat=True)
    blobs = reader.get_blobs(flat=True, discard=discard)

    if filter_log_likelihood:
        distribution_plot = eval(f" distribution[{filter_log_likelihood}]")
    else:
        distribution_plot = distribution

    distribution_plot = distribution_plot.reshape(-1, ndim)

    # Create subplots
    fig = make_subplots(
        rows=ndim, cols=1, subplot_titles=labels, vertical_spacing=0.03
    )

    colors = px.colors.qualitative.Set1

    for i in range(ndim):
        fig.add_trace(
            go.Histogram(
                x=distribution_plot[:, i],
                nbinsx=200,
                name=labels[i],
                marker_color=colors[i % len(colors)],
                showlegend=False,
                hovertemplate=f"{labels[i]}<br>Value: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )

        # Set x-axis range
        if min_bounds_list[i] is not None and max_bounds_list[i] is not None:
            fig.update_xaxes(
                range=[min_bounds_list[i], max_bounds_list[i]],
                row=i + 1,
                col=1,
            )

        fig.update_yaxes(title_text="Count", row=i + 1, col=1)

    fig.update_layout(
        height=200 * ndim,
        title_text=f"📊 Parameter Distributions - {csv_name.split('/')[-1]}",
        title_x=0.5,
    )

    return fig


def plot_lifetime_interactive(
    reader,
    range_chi_square=(7, 11),
    range_log_prior=(7, 11),
    discard=5,
    temperature=300,
    filter_log_likelihood=False,
):
    """Interactive plot of lifetime distribution."""
    print("number of iterations", reader.iteration)
    blobs = reader.get_blobs(flat=True, discard=discard)

    if filter_log_likelihood:
        blobs = blobs[
            blobs["log_likelihood_spectra"]
            > max(blobs["log_likelihood_spectra"]) * 3
        ]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Log k_nr at {temperature} K",
            f"Log k_r at {temperature} K",
            "Log PL QE",
            "Lifetime (nanoseconds)",
        ],
    )

    # Log k_nr
    fig.add_trace(
        go.Histogram(
            x=np.log10(blobs[f"Ex_knr_{temperature:.1f}K"]),
            nbinsx=30,
            name="Log k_nr",
            marker_color="blue",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Log k_r
    fig.add_trace(
        go.Histogram(
            x=np.log10(blobs[f"Ex_kr_{temperature:.1f}K"]),
            nbinsx=30,
            name="Log k_r",
            marker_color="green",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # PL QE
    pl_QE = blobs[f"Ex_kr_{temperature:.1f}K"] / (
        blobs[f"Ex_kr_{temperature:.1f}K"]
        + blobs[f"Ex_knr_{temperature:.1f}K"]
    )
    fig.add_trace(
        go.Histogram(
            x=np.log10(pl_QE),
            nbinsx=30,
            name="Log PL QE",
            marker_color="red",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Lifetime
    lifetime = 1 / (
        blobs[f"Ex_kr_{temperature:.1f}K"]
        + blobs[f"Ex_knr_{temperature:.1f}K"]
    )
    fig.add_trace(
        go.Histogram(
            x=lifetime * 1e9,
            nbinsx=30,
            name="Lifetime",
            marker_color="orange",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_xaxes(
        title_text=f"Log k_nr at {temperature} K",
        row=1,
        col=1,
        range=range_chi_square,
    )
    fig.update_xaxes(
        title_text=f"Log k_r at {temperature} K",
        row=1,
        col=2,
        range=range_log_prior,
    )
    fig.update_xaxes(title_text="Log PL QE", row=2, col=1, range=(-4, 0))
    fig.update_xaxes(title_text="Lifetime (nanoseconds)", row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(title_text="Number of Samples", row=i, col=j)

    fig.update_layout(
        height=600, title_text="⏱️ Lifetime Analysis", title_x=0.5
    )

    return fig


def plot_blobs_chain_interactive(
    reader,
    discard=0,
    temp_lifetime=[60, 140, 220, 300],
    filter_log_likelihood="",
):
    """Interactive plot of blob chains with lifetime calculations."""
    blobs = reader.get_blobs(discard=discard)
    num_blobs = len(blobs.dtype.names)

    # Create subplots for blobs + lifetime
    fig = make_subplots(
        rows=num_blobs + 1,
        cols=1,
        subplot_titles=list(blobs.dtype.names) + ["Lifetime (ns)"],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    colors = px.colors.qualitative.Set3
    colors = [c for c in px.colors.qualitative.Set3 if c.lower() not in ['#ffffb3', '#ffffb3'.lower(), 'rgb(255, 255, 179)', 'rgb(255,255,179)']]

    # Plot blob chains
    for i in range(num_blobs):
        blob_name = blobs.dtype.names[i]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(blobs[blob_name]))),
                y=blobs[blob_name][0],
                mode="lines",
                name=blob_name,
                line=dict(color=colors[i % len(colors)], width=3),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f"{blob_name}<br>Step: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=blob_name, row=i + 1, col=1, tickformat=".2e")

    # Calculate and plot lifetimes
    lifetime_dict = {}
    for temp in temp_lifetime:
        try:
            lifetime_dict[temp] = (
                1 / (blobs[f"Ex_kr_{temp}K"] + blobs[f"Ex_knr_{temp}K"]) * 1e9
            )
        except (ValueError, KeyError):
            print(f"Warning: Keys for temperature {temp}K not found in blobs")

    # Plot lifetimes
    for i, (temp, lifetime) in enumerate(lifetime_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(lifetime))),
                y=lifetime,
                mode="lines",
                name=f"{temp} K",
                line=dict(
                    color=colors[(i + num_blobs) % len(colors)], width=1
                ),
                opacity=0.7,
                hovertemplate=f"{temp} K<br>Step: %{{x}}<br>Lifetime: %{{y:.2f}} ns<extra></extra>",
            ),
            row=num_blobs + 1,
            col=1,
        )

    fig.update_yaxes(title_text="Lifetime (ns)", row=num_blobs + 1, col=1,tickformat=".2f")
    fig.update_xaxes(title_text="Step Number", row=num_blobs + 1, col=1,tickformat=".2f")

    fig.update_layout(
        height=150 * (num_blobs + 1),
        title_text="🧬 Blob Chains and Lifetime Analysis",
        title_x=0.5,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return fig


def plot_corner_interactive_enhanced(
    reader,
    model_config_save,
    discard=10,
    filter_log_likelihood=False,
    **kwargs,
):
    """Create an interactive corner plot similar to sns.pairplot with enhanced features.

    Performance optimized version with data subsampling and reduced trace complexity.

    Parameters
    ----------
    reader : emcee reader object
        MCMC chain reader
    model_config_save : dict
        Model configuration
    discard : int
        Number of initial samples to discard
    filter_log_likelihood : bool or str
        Filter condition for log likelihood
    **kwargs : dict
        Additional options:
        - color_by_likelihood : bool (default True)
        - show_contours : bool (default False - disabled for speed)
        - bins : int (default 30 - reduced for speed)
        - max_points : int (default 5000 - subsample for large datasets)

    """
    # Extract options from kwargs with performance defaults
    color_by_likelihood = kwargs.get("color_by_likelihood", True)
    show_contours = kwargs.get(
        "show_contours", True
    )  # Disabled by default for speed
    bins = kwargs.get("bins", 30)  # Reduced default for speed
    max_points = kwargs.get("max_points", 5000)  # Subsample large datasets

    csv_name = model_config_save["csv_name_pl"]

    # Prepare parameter labels efficiently
    labels = []
    for key in model_config_save["params_to_fit_init"]:
        labels.extend(
            [
                f"{key}_{x}"
                for x in model_config_save["params_to_fit_init"][key]
            ]
        )

    # Get data efficiently
    samples = reader.get_chain(discard=discard, flat=True)

    # Apply filtering if specified
    if filter_log_likelihood:
        samples = eval(f" samples[{filter_log_likelihood}]")

    samples = samples.reshape(-1, len(labels))

    # Subsample for performance if dataset is large
    n_samples = len(samples)
    if n_samples > max_points:
        # Random subsample to maintain distribution characteristics
        idx = np.random.choice(n_samples, max_points, replace=False)
        samples = samples[idx]
        print(
            f"Subsampled from {n_samples} to {max_points} points for better performance"
        )

    # Create DataFrame efficiently (only for required columns)
    df_samples = pd.DataFrame(samples, columns=labels)

    # Handle log likelihood coloring efficiently
    likelihood_available = False
    if color_by_likelihood:
        try:
            blobs = reader.get_blobs(flat=True, discard=discard)
            if hasattr(blobs, "log_likelihood_spectra"):
                likelihood_data = blobs["log_likelihood_spectra"]
                if filter_log_likelihood:
                    likelihood_data = likelihood_data[
                        eval(filter_log_likelihood)
                    ]

                # Subsample likelihood data to match samples
                if (
                    n_samples > max_points
                    and len(likelihood_data) == n_samples
                ):
                    likelihood_data = likelihood_data[idx]
                elif len(likelihood_data) == len(df_samples):
                    pass  # Already correct size
                else:
                    color_by_likelihood = False

                if color_by_likelihood and len(likelihood_data) == len(
                    df_samples
                ):
                    df_samples["log_likelihood"] = likelihood_data
                    likelihood_available = True
        except Exception:
            color_by_likelihood = False

    n_params = len(labels)

    # Use scatter_matrix for faster rendering when possible
    if n_params <= 6 and not show_contours and not likelihood_available:
        # Use the faster px.scatter_matrix for simple cases
        fig = px.scatter_matrix(
            df_samples,
            dimensions=labels,
            title=f"🔄 Parameter Correlations (Fast) - {csv_name.split('/')[-1]}",
            height=min(800, 120 * n_params),
            opacity=0.6,
        )

        # Update diagonal to show histograms
        fig.update_traces(diagonal_visible=False)
        for i, label in enumerate(labels):
            fig.add_trace(
                go.Histogram(
                    x=df_samples[label],
                    name=f"{label}",
                    showlegend=False,
                    opacity=0.7,
                    nbinsx=bins,
                    marker_color="steelblue",
                )
            )

        fig.update_layout(
            title_x=0.5,
            dragmode="select",
            hovermode="closest",
            showlegend=False,
        )
        return fig

    # For complex cases, use optimized subplots approach
    fig = make_subplots(
        rows=n_params,
        cols=n_params,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.03,  # Slightly more spacing for readability
        horizontal_spacing=0.03,
        subplot_titles=None,
    )

    # Pre-calculate correlations to avoid repeated computation
    corr_matrix = df_samples[labels].corr()

    # Configure marker styling once
    if likelihood_available:
        marker_config = {
            "color": df_samples["log_likelihood"],
            "colorscale": "Viridis",
            "size": 2,  # Smaller markers for better performance
            "opacity": 0.7,
            "showscale": True,
            "colorbar": {
                "title": "Log Likelihood",
                "x": 1.02,
                "len": 0.3,
                "y": 0.7,
            },
        }
    else:
        marker_config = {"color": "rgba(31, 119, 180, 0.7)", "size": 2}

    # Populate subplot matrix efficiently
    for i in range(n_params):
        for j in range(n_params):
            row_idx, col_idx = i + 1, j + 1

            if i == j:
                # Diagonal: fast histograms
                fig.add_trace(
                    go.Histogram(
                        x=df_samples[labels[i]],
                        nbinsx=bins,
                        showlegend=False,
                        opacity=0.7,
                        marker_color="steelblue",
                        histnorm="probability density",
                        hoverinfo="x+y",  # Simplified hover
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                # Add mean line (faster than vline)
                mean_val = df_samples[labels[i]].mean()
                y_range = [
                    0,
                    df_samples[labels[i]].max() * 0.1,
                ]  # Approximate height
                fig.add_trace(
                    go.Scatter(
                        x=[mean_val, mean_val],
                        y=y_range,
                        mode="lines",
                        line={"color": "red", "dash": "dash", "width": 2},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            elif i > j:
                # Lower triangle: optimized scatter plots
                fig.add_trace(
                    go.Scattergl(  # Use WebGL for better performance
                        x=df_samples[labels[j]],
                        y=df_samples[labels[i]],
                        mode="markers",
                        marker=marker_config,
                        showlegend=False,
                        hovertemplate=f"{labels[j]}: %{{x:.3f}}<br>{labels[i]}: %{{y:.3f}}<extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                # Add correlation annotation (using pre-calculated values)
                corr = corr_matrix.loc[labels[i], labels[j]]
                fig.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref=f"x{i*n_params + j + 1} domain",
                    yref=f"y{i*n_params + j + 1} domain",
                    text=f"r={corr:.2f}",  # Reduced precision
                    showarrow=False,
                    font={"size": 9, "color": "black"},
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray",
                    borderwidth=1,
                )

    # Batch update axes for better performance
    axis_updates = {
        "showgrid": True,
        "gridwidth": 0.5,
        "gridcolor": "lightgray",
        "showline": True,
        "linewidth": 1,
        "linecolor": "gray",
        "mirror": False,  # Simplified for speed
    }

    # Apply axis labels and styling efficiently
    for i in range(n_params):
        for j in range(n_params):
            row_idx, col_idx = i + 1, j + 1

            # Update axes in batch
            fig.update_xaxes(
                title_text=labels[j] if i == n_params - 1 else None,
                showticklabels=i == n_params - 1,
                **axis_updates,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=labels[i] if j == 0 else None,
                showticklabels=j == 0,
                **axis_updates,
                row=row_idx,
                col=col_idx,
            )

    # Optimized layout
    fig.update_layout(
        title=f"🔄 Parameter Correlations (Optimized) - {csv_name.split('/')[-1]}",
        title_x=0.5,
        height=min(800, 120 * n_params),  # Cap maximum height
        width=min(800, 120 * n_params),  # Cap maximum width
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        # Performance optimizations
        hovermode="closest",
        dragmode="pan",  # Changed from select for better performance
    )

    return fig


# Keep the original function for backward compatibility
def plot_corner_interactive(
    reader, model_config_save, discard=10, filter_log_likelihood=False
):
    """Interactive corner plot using plotly scatter matrix (original version)."""
    return plot_corner_interactive_enhanced(
        reader, model_config_save, discard, filter_log_likelihood
    )


def plot_corner_fast(
    reader,
    model_config_save,
    discard=10,
    filter_log_likelihood=False,
    max_points=3000,
):
    """Ultra-fast corner plot optimized for large datasets.

    This version prioritizes speed over visual features:
    - Uses px.scatter_matrix for maximum performance
    - Limited customization options
    - Automatic data subsampling
    - Optimized for datasets with many parameters/samples
    """
    # Get parameter labels efficiently
    labels = []
    for key in model_config_save["params_to_fit_init"]:
        labels.extend(
            [
                f"{key}_{x}"
                for x in model_config_save["params_to_fit_init"][key]
            ]
        )

    # Get and process data
    samples = reader.get_chain(discard=discard, flat=True)
    if filter_log_likelihood:
        samples = eval(f" samples[{filter_log_likelihood}]")

    samples = samples.reshape(-1, len(labels))

    # Aggressive subsampling for speed
    n_samples = len(samples)
    if n_samples > max_points:
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        idx = rng.choice(n_samples, max_points, replace=False)
        samples = samples[idx]

    # Create DataFrame and plot
    df_samples = pd.DataFrame(samples, columns=labels)
    csv_name = model_config_save["csv_name_pl"]

    # Use the fastest plotting method
    fig = px.scatter_matrix(
        df_samples,
        dimensions=labels,
        title=f"🚀 Fast Parameter Correlations - {csv_name.split('/')[-1]}",
        height=min(600, 100 * len(labels)),
        opacity=0.6,
    )

    # Minimal styling for maximum speed
    fig.update_layout(title_x=0.5, showlegend=False, dragmode="pan")

    return fig


def convert_matplotlib_to_plotly(fig_matplotlib):
    """Helper function to convert matplotlib figures to plotly if needed."""
    # This is a placeholder - in practice, you might want to use plotly.tools.mpl_to_plotly
    # or recreate the plot directly in plotly for better interactivity


def plot_fit_limits_overlay_interactive(model_config, model_config_save):
    """Interactive version of fit limits overlay plot using Plotly."""
    try:
        # Import required modules
        from pl_temp_fit import Exp_data_utils, config_utils, covariance_utils
        from pl_temp_fit.model_function import LTL

        # Get parameters
        fixed_parameters_dict, params_to_fit, min_bound, max_bound = (
            config_utils.get_dict_params(model_config_save)
        )
        csv_name = model_config_save["csv_name_pl"]
        exp_data, temperature_list, hws = Exp_data_utils.read_data(csv_name)
        save_folder = model_config_save["save_folder"]

        # Get covariance data
        co_var_mat_pl, variance_pl = covariance_utils.plot_generated_data_pl(
            save_folder,
            model_config,
            savefig=False,
            fixed_parameters_dict=fixed_parameters_dict,
            params_to_fit=model_config_save["params_to_fit_init"],
        )

        # Parameter sets to compare
        param_sets = [
            (
                model_config_save["params_to_fit_init"],
                "Initial",
                "#1f77b4",
                "solid",
            ),
            (model_config_save["min_bounds"], "Min Bound", "#ff7f0e", "dash"),
            (model_config_save["max_bounds"], "Max Bound", "#2ca02c", "dot"),
        ]

        # Helper function for PL calculation
        def pl_trial(
            temperature_list_pl,
            hws_pl,
            fixed_parameters_dict={},
            params_to_fit={},
        ):
            data = LTL.Data()
            data.update(**fixed_parameters_dict)
            data.update(**params_to_fit)
            data.D.Luminecence_exp = "PL"
            data.D.T = temperature_list_pl
            LTL.ltlcalc(data)
            pl_results = data.D.kr_hw
            pl_results_interp = np.zeros(
                (len(hws_pl), len(temperature_list_pl))
            )

            for i in range(len(temperature_list_pl)):
                pl_results_interp[:, i] = np.interp(
                    hws_pl, data.D.hw, pl_results[:, i]
                )

            pl_results_interp = (
                pl_results_interp
                / pl_results_interp[pl_results_interp > 0].max()
            )
            return pl_results_interp

        # Calculate model results for each parameter set
        model_results = []
        for params, label, color, style in param_sets:
            pl_results = pl_trial(
                model_config["temperature_list_pl"],
                model_config["hws_pl"],
                fixed_parameters_dict,
                params,
            )
            model_results.append((pl_results, label, color, style))

        # Create subplots for each temperature
        n_temps = len(temperature_list)
        if n_temps == 0:
            raise ValueError("No temperature data found")

        # Format subplot titles properly
        subplot_titles = []
        for temp in temperature_list:
            if isinstance(temp, (int, float)):
                subplot_titles.append(f"{temp:.0f} K")
            else:
                # Handle string format like "300K" or "300.0K"
                temp_str = str(temp)
                if "K" in temp_str:
                    temp_val = float(temp_str.replace("K", ""))
                    subplot_titles.append(f"{temp_val:.0f} K")
                else:
                    subplot_titles.append(f"{temp} K")

        fig = make_subplots(
            rows=(n_temps + 1) // 2,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        # Color palette for experimental data
        exp_colors = px.colors.qualitative.Set1

        # Ensure experimental data has the right shape
        if exp_data.shape[1] != len(temperature_list):
            raise ValueError(
                f"Experimental data shape {exp_data.shape} doesn't match temperature list length {len(temperature_list)}"
            )

        y_max = np.max(exp_data[:, :]) if exp_data.size > 0 else 1
        # Plot each temperature
        for i, temp in enumerate(temperature_list):
            row = (i // 2) + 1
            col = (i % 2) + 1

            # Plot experimental data with error bars
            y_exp = exp_data[:, i]
            y_err = np.sqrt(variance_pl[:, i])

            # Normalize experimental data to match model scale (0-1)
            # Normalize each temperature individually to show relative shape

            if y_max > 0:
                y_exp_normalized = y_exp / y_max
                y_err_normalized = y_err
            else:
                y_exp_normalized = y_exp
                y_err_normalized = y_err

            # Experimental data
            fig.add_trace(
                go.Scatter(
                    x=hws,
                    y=y_exp_normalized,
                    error_y={
                        "type": "data",
                        "array": y_err_normalized,
                        "visible": True,
                    },
                    mode="markers",
                    name=f"Exp {temp} K",
                    marker={
                        "color": exp_colors[i % len(exp_colors)],
                        "size": 6,
                    },
                    legendgroup=f"exp_{i}",
                    showlegend=i == 0,
                    hovertemplate=(
                        f"<b>Experimental - {temp} K</b><br>"
                        "Energy: %{x:.3f} eV<br>"
                        "PL Intensity: %{y:.3f}<br>"
                        "Error: ±%{error_y.array:.3f}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            # Plot model predictions for each parameter set
            for j, (pl_results, label, color, style) in enumerate(
                model_results
            ):
                line_style = {"dash": "dash", "dot": "dot", "solid": "solid"}[
                    style
                ]

                fig.add_trace(
                    go.Scatter(
                        x=model_config["hws_pl"],
                        y=pl_results[:, i],
                        mode="lines",
                        name=f"PL {label}",
                        line={"color": color, "dash": line_style, "width": 2},
                        legendgroup=f"model_{j}",
                        showlegend=i == 0,
                        hovertemplate=(
                            f"<b>PL {label} - {temp} K</b><br>"
                            "Energy: %{x:.3f} eV<br>"
                            "Normalized PL: %{y:.3f}<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

            # Update y-axis range for this subplot
            fig.update_yaxes(range=[0, 1.1], row=row, col=col)

        # Update layout
        fig.update_layout(
            title="🔍 PL Fit Limits Overlay (Interactive)",
            title_x=0.5,
            height=300 * ((n_temps + 1) // 2),
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.08,
                "xanchor": "center",
                "x": 0.5,
            },
        )

        # Update all axes
        fig.update_xaxes(
            title_text="Energy (eV)",
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="Normalized PL Intensity",
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )

        return fig

    except Exception as e:
        # Fallback: create a simple error plot
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Error creating interactive plot: {e!s}",
            showarrow=False,
            xref="paper",
            yref="paper",
            font={"size": 16, "color": "red"},
        )
        fig.update_layout(
            title="❌ Error in Fit Limits Plot",
            height=400,
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig


def _calculate_transfer_rates(
    params: dict, temperatures: np.ndarray, kb: float
) -> tuple:
    """Calculate transfer rates for Arrhenius plot."""
    if "D" not in params or "log_kEXCT" not in params["D"]:
        return None, None, 0

    d_params = params["D"]
    log_k_ex_ct = d_params["log_kEXCT"]
    k_ex_ct = 10**log_k_ex_ct

    # Calculate activation energy from state energies
    activation_energy = 0
    if "EX" in params and "CT" in params:
        ex_energy = params["EX"].get("E", 0)
        ct_energy = params["CT"].get("E", 0)
        activation_energy = max(0, ex_energy - ct_energy)

    # Calculate transfer rates
    if activation_energy > 0:
        ln10 = np.log(10)
        log_k_transfer = log_k_ex_ct - activation_energy / (
            kb * temperatures * ln10
        )
    else:
        log_k_transfer = [log_k_ex_ct] * len(temperatures)

    return log_k_transfer, k_ex_ct, activation_energy


def _add_transfer_trace(fig, params, inv_t, temperatures, colors) -> bool:
    """Add transfer rate trace to Arrhenius plot."""
    kb = 8.617e-5  # Boltzmann constant in eV/K
    log_k_transfer, k_ex_ct, activation_energy = _calculate_transfer_rates(
        params, temperatures, kb
    )

    if log_k_transfer is None:
        return False

    d_params = params["D"]
    fig.add_trace(
        go.Scatter(
            x=inv_t,
            y=log_k_transfer,
            mode="lines",
            name="Ex→CT Transfer Rate",
            line={"color": colors["Transfer"], "width": 3, "dash": "solid"},
            hovertemplate=(
                f"<b>Ex→CT Transfer</b><br>"
                f"Status: ON<br>"
                f"Activation Energy: {activation_energy:.3f} eV<br>"
                f"Rate Constant: {k_ex_ct:.2e} s⁻¹<br>"
                f"Temperature: %{{customdata:.1f}} K<br>"
                f"Transfer Rate: 10^%{{y:.2f}} s⁻¹<br>"
                f"Additional properties:<br>"
                f"• RCTE: {d_params.get('RCTE', 'N/A')}<br>"
                "<extra></extra>"
            ),
            customdata=temperatures,
            yaxis="y2",
            showlegend=True,
        )
    )
    return True


def _add_energy_gap_annotation(fig, params, inv_t) -> None:
    """Add energy gap annotation and shading."""
    if "EX" not in params or "CT" not in params:
        return

    ex_energy = params["EX"].get("E", 0)
    ct_energy = params["CT"].get("E", 0)
    energy_gap = abs(ex_energy - ct_energy)

    if energy_gap <= 0:
        return

    # Add shaded region
    fig.add_shape(
        type="rect",
        x0=min(inv_t),
        x1=max(inv_t),
        y0=min(ex_energy, ct_energy),
        y1=max(ex_energy, ct_energy),
        fillcolor="rgba(128, 128, 128, 0.2)",
        line={"width": 0},
        layer="below",
    )

    # Add annotation
    fig.add_annotation(
        x=np.mean(inv_t),
        y=(ex_energy + ct_energy) / 2,
        text=f"ΔE = {energy_gap:.3f} eV",
        showarrow=True,
        arrowhead=2,
        arrowcolor="gray",
        font={"size": 12, "color": "gray"},
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
    )


def _create_ground_state_trace(state_positions, colors):
    """Create ground state trace for state diagram."""
    ground_energy = 0.0
    return go.Scatter(
        x=[state_positions["Ground"] - 0.4, state_positions["Ground"] + 0.4],
        y=[ground_energy, ground_energy],
        mode="lines",
        name="Ground State",
        line={"color": colors["Ground"], "width": 6},
        hovertemplate=(
            "<b>Ground State</b><br>"
            "Energy: 0.000 eV<br>"
            "Status: ON<br>"
            "Reference state<br>"
            "<extra></extra>"
        ),
        showlegend=True,
    )


def _create_exciton_state_trace(params, state_positions, colors):
    """Create exciton state trace if parameters exist."""
    if "EX" not in params or "E" not in params["EX"]:
        return None, None

    ex_params = params["EX"]
    ex_energy = ex_params["E"]
    ex_enabled = not ex_params.get("off", 0)
    # print(f"Exciton energy: {ex_energy}, enabled: {ex_enabled}")
    trace = go.Scatter(
        x=[state_positions["Exciton"] - 0.4, state_positions["Exciton"] + 0.4],
        y=[ex_energy, ex_energy],
        mode="lines",
        name="Exciton State",
        line={
            "color": colors["Exciton"],
            "width": 6,
            "dash": "solid" if ex_enabled else "dash",
        },
        hovertemplate=(
            f"<b>Exciton State</b><br>"
            f"Energy: {ex_energy:.3f} eV<br>"
            f"Status: {'ON' if ex_enabled else 'OFF'}<br>"
            f"Properties:<br>"
            f"• sigma: {ex_params.get('sigma', 'N/A')}<br>"
            f"• Li: {ex_params.get('Li', 'N/A')}<br>"
            f"• Lo: {ex_params.get('Lo', 'N/A')}<br>"
            f"• hO: {ex_params.get('hO', 'N/A')}<br>"
            "<extra></extra>"
        ),
        showlegend=True,
    )

    state_dict = {
        "name": "Exciton",
        "energy": ex_energy,
        "enabled": ex_enabled,
        "params": ex_params,
    }
    return trace, state_dict


def _create_ct_state_trace(params, state_positions, colors):
    """Create CT state trace if parameters exist."""
    if "CT" not in params or "E" not in params["CT"]:
        # print(f"CT energy: {params} does not exist in parameters.")

        return None, None

    ct_params = params["CT"]
    ct_energy = ct_params["E"]
    ct_enabled = not ct_params.get("off", 0)
    print(f"CT energy: {ct_energy}, enabled: {ct_enabled}")
    trace = go.Scatter(
        x=[state_positions["CT"] - 0.4, state_positions["CT"] + 0.4],
        y=[ct_energy, ct_energy],
        mode="lines",
        name="CT State",
        line={
            "color": colors["CT"],
            "width": 6,
            "dash": "solid" if ct_enabled else "dash",
        },
        hovertemplate=(
            f"<b>CT State</b><br>"
            f"Energy: {ct_energy:.3f} eV<br>"
            f"Status: {'ON' if ct_enabled else 'OFF'}<br>"
            f"Properties:<br>"
            f"• sigma: {ct_params.get('sigma', 'N/A')}<br>"
            f"• Li: {ct_params.get('Li', 'N/A')}<br>"
            f"• Lo: {ct_params.get('Lo', 'N/A')}<br>"
            f"• hO: {ct_params.get('hO', 'N/A')}<br>"
            f"• log_fosc: {ct_params.get('log_fosc', 'N/A')}<br>"
            "<extra></extra>"
        ),
        showlegend=True,
    )

    state_dict = {
        "name": "CT",
        "energy": ct_energy,
        "enabled": ct_enabled,
        "params": ct_params,
    }
    return trace, state_dict


def _calculate_radiative_rate(ex_state):
    """Calculate radiative recombination rate for exciton state."""
    if "kr" in ex_state["params"]:
        return ex_state["params"]["kr"]
    if "log_kr" in ex_state["params"]:
        return 10 ** ex_state["params"]["log_kr"]

    # Estimate from oscillator strength and energy
    log_fosc = ex_state["params"].get("log_fosc", -2)  # Default value
    fosc = 10**log_fosc
    # Simplified estimate: kr ≈ 10^8 * E² * f [s⁻¹] where E is in eV
    return 1e8 * (ex_state["energy"] ** 2) * fosc


def _calculate_nonradiative_rate(ex_state, kr_value):
    """Calculate non-radiative recombination rate for exciton state."""
    if "knr" in ex_state["params"]:
        return ex_state["params"]["knr"]
    if "log_knr" in ex_state["params"]:
        return 10 ** ex_state["params"]["log_knr"]

    # Estimate from total lifetime and radiative rate
    total_rate = 1e7  # Default estimate
    return max(0, total_rate - kr_value)


def _add_transfer_arrow(
    fig, ex_state, ct_state, params, state_positions, colors
):
    """Add transfer arrow between exciton and CT states."""
    if not (ex_state and ct_state and "D" in params):
        return

    d_params = params["D"]
    if "log_kEXCT" not in d_params:
        return

    log_k_ex_ct = d_params["log_kEXCT"]
    k_ex_ct = 10**log_k_ex_ct

    # Calculate temperature-dependent rate at 300K
    temperature_300k = 300.0  # Kelvin
    kb = 8.617e-5  # Boltzmann constant in eV/K

    if ex_state["energy"] > ct_state["energy"]:
        k_300k = k_ex_ct  # Downhill transfer
    else:
        # Uphill transfer (thermally activated)
        activation_energy = abs(ct_state["energy"] - ex_state["energy"])
        k_300k = k_ex_ct * np.exp(-activation_energy / (kb * temperature_300k))

    arrow_offset = 0.15
    fig.add_annotation(
        x=state_positions["CT"] - 0.2,
        y=ct_state["energy"] + arrow_offset,
        ax=state_positions["Exciton"] + 0.2,
        ay=ex_state["energy"] + arrow_offset,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor=colors["Transfer"],
        text=f"k_EXCT(300K)<br>{k_300k:.1e} s⁻¹",
        textangle=0,
        font={"size": 9, "color": colors["Transfer"]},
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=colors["Transfer"],
        borderwidth=1,
    )


def _add_radiative_arrow(fig, ex_state, state_positions, colors):
    """Add radiative recombination arrow from exciton to ground."""
    if not ex_state:
        return

    kr_value = ex_state.kr[0]
    arrow_offset = 0.15
    fig.add_annotation(
        x=state_positions["Ground"] + 0.2,
        y=0.3,  # ground_energy + 0.3
        ax=state_positions["Exciton"] - 0.2,
        ay=ex_state.E - arrow_offset,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor=colors["Radiative"],
        text=f"k_r(300K)<br>{kr_value:.1e} s⁻¹",
        textangle=-30,
        font={"size": 9, "color": colors["Radiative"]},
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=colors["Radiative"],
        borderwidth=1,
    )


def _add_nonradiative_arrow(fig, ex_state, state_positions, colors):
    """Add non-radiative recombination arrow from exciton to ground."""
    if not ex_state:
        return

    knr_value = ex_state.knr[0][0]
    arrow_offset = 0.15
    fig.add_annotation(
        x=state_positions["Ground"] + 0.1,
        y=0.1,  # ground_energy + 0.1
        ax=state_positions["Exciton"] - 0.1,
        ay=ex_state.E - 2 * arrow_offset,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor=colors["NonRadiative"],
        text=f"k_nr(300K)<br>{knr_value:.1e} s⁻¹",
        textangle=-45,
        font={"size": 9, "color": colors["NonRadiative"]},
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=colors["NonRadiative"],
        borderwidth=1,
    )


def _add_ct_recombination_arrow(fig, ct_state, state_positions, colors):
    """Add CT to ground recombination arrow."""
    if not ct_state:
        return

    ct_kr = 1e4  # Typical CT radiative rate
    arrow_offset = 0.15

    fig.add_annotation(
        x=state_positions["Ground"] + 0.3,
        y=0.2,  # ground_energy + 0.2
        ax=state_positions["CT"] - 0.3,
        ay=ct_state["energy"] - arrow_offset,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=colors["Radiative"],
        text=f"CT k_r<br>{ct_kr:.1e} s⁻¹",
        textangle=-60,
        font={"size": 8, "color": colors["Radiative"]},
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=colors["Radiative"],
        borderwidth=1,
    )


def plot_state_diagram_interactive(model_config_save):
    """Create interactive state diagram showing energy levels and transitions.

    Parameters
    ----------
    model_config_save : dict
        Model configuration containing parameter values

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive state diagram plot

    """
    #try:
    from pl_temp_fit.model_function import LTL

    params = model_config_save.get("params_to_fit_init", {})
    fixed_params = model_config_save.get("fixed_parameters_dict", {})
    data = LTL.Data()
    data.update(**fixed_params)
    data.update(**params)
    data.D.Luminecence_exp = "PL"
    data.D.T = np.array([300.0])
    LTL.ltlcalc(data)

    # merge fixed parameters into params
    for key, value in fixed_params.items():
        if key not in params:
            params[key] = value
        if isinstance(value, dict):
            params[key].update(value)

    fig = go.Figure()
    colors = {
        "Exciton": "#1f77b4",  # Blue
        "CT": "#ff7f0e",  # Orange
        "Ground": "#2ca02c",  # Green
        "Transfer": "#d62728",  # Red
        "Radiative": "#9467bd",  # Purple
        "NonRadiative": "#8c564b",  # Brown
    }

    state_positions = {"Ground": 0, "Exciton": 2, "CT": 4}
    min_states = 2

    # Add ground state
    fig.add_trace(_create_ground_state_trace(state_positions, colors))

    # Add state energy levels
    state_info = []

    # Exciton state
    ex_trace, ex_state = _create_exciton_state_trace(
        params, state_positions, colors
    )
    if ex_trace:
        fig.add_trace(ex_trace)
        state_info.append(ex_state)

    # CT state
    ct_trace, ct_state = _create_ct_state_trace(
        params, state_positions, colors
    )
    if ct_trace:
        fig.add_trace(ct_trace)
        state_info.append(ct_state)

    # Add transition arrows
    _add_transfer_arrow(
        fig, ex_state, ct_state, params, state_positions, colors
    )
    _add_radiative_arrow(fig, data.EX, state_positions, colors)
    _add_nonradiative_arrow(fig, data.EX, state_positions, colors)
    # _add_radiative_arrow(fig, data.CT, state_positions, colors)

    # Add energy difference annotation
    if len(state_info) >= min_states:
        energies = [s["energy"] for s in state_info if s["enabled"]]
        if len(energies) >= min_states:
            energy_diff = max(energies) - min(energies)
            max_energy = max(energies)

            fig.add_annotation(
                x=2,  # Center of the diagram
                y=max_energy + 0.4,
                text=f"ΔE = {energy_diff:.3f} eV",
                showarrow=False,
                font={
                    "size": 14,
                    "color": "black",
                    "family": "Arial Black",
                },
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=2,
            )

    # Add legend for arrow types
    legend_y_start = 0.95
    legend_items = [
        ("Ex→CT Transfer", colors["Transfer"]),
        ("Radiative Decay", colors["Radiative"]),
        ("Non-radiative Decay", colors["NonRadiative"]),
    ]

    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=0.02,
            y=legend_y_start - i * 0.05,
            text=f"→ {label}",
            showarrow=False,
            xref="paper",
            yref="paper",
            font={"size": 10, "color": color},
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=color,
            borderwidth=1,
        )

    # Configure layout
    fig.update_layout(
        title="⚡ Electronic State Diagram with Transition Rates (300K)",
        title_x=0.5,
        title_font={"size": 18},
        xaxis={
            "title": "Electronic States",
            "tickmode": "array",
            "tickvals": list(state_positions.values()),
            "ticktext": list(state_positions.keys()),
            "range": [-0.8, 4.8],
            "showgrid": False,
            "showline": True,
            "linewidth": 2,
            "linecolor": "black",
        },
        yaxis={
            "title": "Energy (eV)",
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "lightgray",
            "showline": True,
            "linewidth": 2,
            "linecolor": "black",
            "zeroline": True,
            "zerolinewidth": 3,
            "zerolinecolor": "black",
        },
        height=700,
        width=900,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.85,
            "xanchor": "left",
            "x": 1.02,
        },
        font={"size": 12},
    )
    
    return fig
"""
    except (KeyError, ValueError, TypeError) as e:
        # Fallback: create error plot
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Error creating state diagram: {e!s}",
            showarrow=False,
            xref="paper",
            yref="paper",
            font={"size": 16, "color": "red"},
        )
        fig.update_layout(
            title="❌ Error in State Diagram",
            height=400,
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
    """
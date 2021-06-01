import time

import matplotlib.pyplot as plt
from Results import *
import numpy as np

COLORS = ['tab:red', 'royalblue', 'forestgreen', 'darkorange', 'plum']


def main():
    # calculate_and_plot_divergence(separate=True)

    print("Loading results...")
    t = time.time()
    # res1 = StateResults("results/states/[20-10]_150y_1000a_mu.states")

    # res1 = ErrorResults("results/errors/[10-10]_150y_1000a_mu.errors")
    # res2 = ErrorResults("results/errors/[20-10]_150y_1000a_mu.errors")
    # res3 = ErrorResults("results/errors/[10-20]_150y_1000a_mu.errors")

    res1 = SingleResult("results/single/[10-20]_150y_1a_mu.single")  # TODO: Show prior somehow
    print("Done! ({:.2f} seconds)".format(time.time() - t))

    # plot_errors([res1, res2, res3])
    # plot_state_results(res1)
    plot_simulation(res1)


# def calculate_and_plot_divergence(separate=False):
#     # Get the KL Divergence over time for each layer
#     KLs = [l for l in zip(*[[KLDiv_normal(generated_temps[t][1][i], get_params(agent_params, t)[i])
#                              for i in range(min(len(generated_temps[t][1]), len(agent_params[0])))]
#                             for t in range(N_ITER)])]
#
#     # Time array in terms of years for plotting
#     time = np.arange(0, N_ITER) / YEAR_LEN
#
#     if separate:
#         fig, axs = plt.subplots(1, len(LAYER_STATES), figsize=(10, 5))
#
#         for layer, KL in enumerate(KLs):
#             # axs[layer].scatter(time, KL, s=1, color='k')
#             axs[layer].plot(time, KL, color='k')
#             axs[layer].set_title("Layer {}".format(layer))
#             axs[layer].set_ylim([0, np.array(KLs).max()])
#         fig.supxlabel("Time (years)")
#         fig.supylabel("KL Divergence")
#         fig.suptitle("KL Divergence of the generative process and generative model")
#     else:
#         for layer, KL in reversed(list(enumerate(KLs))):
#             print(layer)
#             plt.plot(time, KL, label="Layer {}".format(layer))
#         plt.title("KL Divergence of the generative process and generative model")
#         plt.xlabel("Time (years)")
#         plt.ylabel("KL Divergence")
#     plt.show()


# # TODO: Incorporate standard deviation into model error?
# def model_error(gen_temps, params):
#     temps = [t[0][0]["value"] for t in gen_temps]
#     mu_sums = [sum(p["mu"] for p in get_params(params, t)) for t in range(len(params))]
#     model_error = [abs(temps[t] - mu_sums[t]) for t in range(len(temps))]
#     return model_error


def plot_errors(results):
    all_errors = np.array([result.model_errors for result in results])
    max = np.max(all_errors)

    fig, axs = plt.subplots(1, len(results), figsize=(3 * (len(results) + 1), 6))
    for i, result in enumerate(results):
        errors = np.array(result.model_errors).T
        axs[i].plot(np.arange(result.N_ITER)/result.YEAR_LEN, errors, COLORS[i], alpha=0.01, zorder=-1)
        axs[i].set_ylim((0, max))

        end_max = np.max(errors[-result.YEAR_LEN:, :])
        axs[i].text(x=100, y=end_max + 8, s="ε = {:.2f}".format(end_max))
        axs[i].hlines(y=end_max, xmin=0, xmax=result.N_ITER/result.YEAR_LEN, colors=['black'], zorder=1, linewidths=[2], linestyle='dashed', label="Max resulting error")

        axs[i].set_title("Layer states: {}".format(result.LAYER_STATES))
        axs[i].set_xlabel("Time (years)")
        axs[i].set_ylabel("Model error (ε)")
        axs[i].legend(loc="upper right")
    plt.suptitle("Model errors of {} agents with random priors".format(len(results[0].model_errors)))
    plt.tight_layout()
    plt.show()


def plot_state_results(results):
    layer_count = len(results.LAYER_STATES)
    time_vector = np.arange(results.N_ITER)/results.YEAR_LEN

    all_sigs = np.array([params[2] for params in results.agent_params])
    sig_conv = np.max(np.max(all_sigs[:, -2*results.YEAR_LEN:, :], axis=0), axis=0)
    sig_max = np.max(all_sigs)
    del all_sigs

    all_eqs = np.array([params[1] for params in results.agent_params])
    eqs_min = np.min(all_eqs)
    eqs_max = np.max(all_eqs)
    del all_eqs

    fig, axs = plt.subplots(3, layer_count + 1, figsize=(5*(layer_count + 1), 10))
    eqs_sum_sum = []
    for i, params in enumerate(results.agent_params):
        amps, eqs, sigs = params
        amps_per_layer = [[m[i] for m in amps] for i in range(len(amps[0]))]
        eqs_per_layer = [[e[i] for e in eqs] for i in range(len(eqs[0]))]
        sigs_per_layer = [[s[i] for s in sigs] for i in range(len(sigs[0]))]

        eqs_sum = [sum(e) for e in eqs]
        eqs_sum_sum.append(eqs_sum)

        alpha = 0.01

        if layer_count > 1:
            for layer in range(layer_count):
                axs[0, layer].plot(time_vector, sigs_per_layer[layer], color=COLORS[layer], alpha=alpha, zorder=-1)
                axs[0, layer].set_ylim([0, sig_max])
                axs[1, layer].plot(time_vector, amps_per_layer[layer], color=COLORS[layer], alpha=alpha)
                axs[2, layer].plot(time_vector, eqs_per_layer[layer], color=COLORS[layer], alpha=alpha)
                axs[2, layer].set_ylim([eqs_min, eqs_max])
        else:
            axs[0, 0].plot(time_vector, sigs_per_layer[0], color=COLORS[0], alpha=alpha)
            axs[1, 0].plot(time_vector, amps_per_layer[0], color=COLORS[0], alpha=alpha)
            axs[2, 0].plot(time_vector, eqs_per_layer[0], color=COLORS[0], alpha=alpha)

        axs[2, layer_count].plot(time_vector, eqs_sum, color='k', alpha=alpha, zorder=-1)

    for layer in range(layer_count):
        axs[0, layer].set_title("Standard deviation layer {}".format(layer))
        axs[0, layer].text(x=135, y=sig_conv[layer] + 5, s="{:.2f}".format(sig_conv[layer]))
        axs[0, layer].hlines(y=sig_conv[layer], xmin=0, xmax=results.N_ITER / results.YEAR_LEN, colors=['k'], zorder=1, label="Max resulting standard deviation")
        axs[0, layer].set_xlabel("Time (years)")
        axs[0, layer].set_ylabel("Standard deviation")
        axs[0, layer].legend(loc="upper right")
        axs[1, layer].set_title("Amplitude layer {}".format(layer))
        axs[1, layer].set_xlabel("Time (years)")
        axs[1, layer].set_ylabel("Amplitude")
        axs[2, layer].set_title("Equilibrium layer {}".format(layer))
        axs[2, layer_count].set_title("Sum of equilibria")
        axs[2, layer_count].set_xlabel("Time (years)")
        axs[2, layer_count].set_ylabel("Equilibrium")

    line_y = np.mean(eqs_sum_sum)
    axs[2, layer_count].text(x=135, y=line_y + 10, s="{:.2f}".format(line_y))
    axs[2, layer_count].hlines(y=line_y, xmin=0, xmax=results.N_ITER/results.YEAR_LEN, colors=['r'], zorder=1, label="Mean sum of equilibria")
    axs[2, layer_count].legend(loc="upper right")

    fig.suptitle("Model parameters per layer of {} agents with random priors\nLayer states: {}".format(len(results.agent_params), results.LAYER_STATES))
    plt.tight_layout()
    plt.show()


# def get_params(params, t):
#     state0 = t % LAYER_STATES[0]
#     states = [state0]
#     for i in range(1, len(params[t])):
#         states.append((t // LAYER_STATES[i-1]) % LAYER_STATES[i])
#
#     return [params[t][i][states[i]] for i in range(len(params[t]))]


def plot_simulation(results):
    print("Generating plots...")

    observations = [o[0][0]["value"] for o in results.generated_temps]

    # Plotting the first year of generated and predicted temperatures
    year_time = list(range(results.YEAR_LEN))
    plot_temperatures(year_time,
                      observations[:results.YEAR_LEN], "observations",
                      results.predictions[:results.YEAR_LEN], "predictions",
                      "First year generated VS predicted temperatures\nLayer states: {}".format(results.LAYER_STATES),
                      func="plot")

    # Plotting the last year of generated and predicted temperatures
    plot_temperatures(year_time,
                      observations[-results.YEAR_LEN:], "observations",
                      results.predictions[-results.YEAR_LEN:], "predictions",
                      "Last year generated VS predicted temperatures\nLayer states: {}".format(results.LAYER_STATES),
                      func="plot")

    plot_states(results)

    print("Done!")


def plot_temperatures(time, obs, obs_label, pred, pred_label, title, func="scatter"):
    plt.figure(figsize=(8, 4))
    if func == "scatter":
        plt.scatter(time, obs, color='k', s=10, label=obs_label)
        plt.scatter(time, pred, color='r', s=10, label=pred_label)
    else:
        plt.plot(time, obs, color='k', label=obs_label)
        plt.plot(time, pred, color='r', label=pred_label)
    plt.title(title)
    plt.xlabel("Time (iterations)")
    plt.ylabel("Temperature value")
    plt.legend()
    plt.show()


def plot_states(results):
    params_per_layer = [l for l in zip(*results.agent_params)]

    for layer, params in enumerate(params_per_layer):
        final_params = params[-1]

        if len(final_params) < 15:
            distributions = [np.random.normal(loc=p["mu"], scale=p["sigma"], size=10000) for p in final_params]
            lower = min([min(dist) for dist in distributions])
            upper = max([max(dist) for dist in distributions])
            if len(final_params) > 1:
                fig, axs = plt.subplots(1, len(final_params), figsize=(10, 5))

                for state, state_params in enumerate(final_params):
                    axs[state].hist(distributions[state],
                                    orientation="horizontal",
                                    density=True,
                                    color='k',
                                    bins=30,
                                    range=(lower, upper))
                    axs[state].set_title("State {}\nμ = {:.1f}\nσ = {:.1f}".format(state, state_params["mu"], state_params["sigma"]))
                    axs[state].axhline(distributions[state].mean(), color='r')
                fig.supxlabel("Probability")
                fig.supylabel("Value")
                mus = [p["mu"] for p in final_params]
                fig.suptitle("State parameters for layer {}\nequilibrium = {:.1f}, amplitude = {:.1f}".format(layer, sum(mus)/len(final_params), (max(mus) - min(mus))/2))
            else:
                plt.hist(distributions[0],
                         orientation="horizontal",
                         density=True,
                         color='k',
                         bins=30,
                         range=(lower, upper))
                plt.axhline(distributions[0].mean(), color='r')
                plt.xlabel("Probability")
                plt.ylabel("Value")
                plt.title("State parameters for layer {}\nState: μ = {:.1f},\nσ = {:.1f}"
                          .format(layer, final_params[0]["mu"], final_params[0]["sigma"]))
        else:
            xs = range(len(final_params))
            ys, es = zip(*[(p["mu"], p["sigma"]) for p in final_params])

            err = plt.errorbar(xs, ys, yerr=es, color='k')
            err[-1][0].set_linestyle("dotted")
            plt.xlabel("State")
            plt.ylabel("Value")
            plt.title("State parameters for layer {} with ±σ as error bars\nequilibrium = {:.1f}, amplitude = {:.1f}".format(layer, sum(ys)/len(ys), (max(ys) - min(ys))/2))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

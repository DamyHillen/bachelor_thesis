import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys


filename = "results/28-05-21_23:54:39::711410.results"

print("Loading results from '{}'...".format(filename))
file = open(sys.argv[0] if len(sys.argv) > 1 else filename, "rb")
HOUR_LEN, DAY_LEN, YEAR_LEN, N_ITER, LAYER_STATES, generated_temps, predictions, agent_params = pickle.load(file)
print("Done!")


def main():
    # calculate_and_plot_divergence(separate=True)
    error = model_error(generated_temps, agent_params)
    errors = [[e - k for e in error] for k in range(-10, 10)]

    plot_errors(errors)

    # plot_simulation()


def calculate_and_plot_divergence(separate=False):
    # Get the KL Divergence over time for each layer
    KLs = [l for l in zip(*[[KLDiv_normal(generated_temps[t][1][i], get_params(agent_params, t)[i])
                             for i in range(min(len(generated_temps[t][1]), len(agent_params[0])))]
                            for t in range(N_ITER)])]

    # Time array in terms of years for plotting
    time = np.arange(0, N_ITER) / YEAR_LEN

    if separate:
        fig, axs = plt.subplots(1, len(LAYER_STATES), figsize=(10, 5))

        for layer, KL in enumerate(KLs):
            # axs[layer].scatter(time, KL, s=1, color='k')
            axs[layer].plot(time, KL, color='k')
            axs[layer].set_title("Layer {}".format(layer))
            axs[layer].set_ylim([0, np.array(KLs).max()])
        fig.supxlabel("Time (years)")
        fig.supylabel("KL Divergence")
        fig.suptitle("KL Divergence of the generative process and generative model")
    else:
        for layer, KL in reversed(list(enumerate(KLs))):
            print(layer)
            plt.plot(time, KL, label="Layer {}".format(layer))
        plt.title("KL Divergence of the generative process and generative model")
        plt.xlabel("Time (years)")
        plt.ylabel("KL Divergence")
    plt.show()


# TODO: Incorporate standard deviation into model error?
def model_error(gen_temps, params):
    temps = [t[0][0]["value"] for t in gen_temps]
    mu_sums = [sum(p["mu"] for p in get_params(params, t)) for t in range(len(params))]
    model_error = [abs(temps[t] - mu_sums[t]) for t in range(len(temps))]
    return model_error


def plot_errors(errors):
    for error in errors:
        plt.plot(np.arange(N_ITER)/YEAR_LEN, error, 'r', alpha=0.01)
    plt.xlabel("Time (years)")
    plt.ylabel("Model error")
    plt.show()


def get_params(params, t):
    state0 = (t + 1) % LAYER_STATES[0]
    states = [state0]
    for i in range(1, len(params[t])):
        states.append(((t + 1) // LAYER_STATES[i-1]) % LAYER_STATES[i])
    return [params[t][i][states[i]] for i in range(len(params[t]))]


def KLDiv_normal(params0, params1):
    mu0, sigma0 = params0["mu"], params0["sigma"]
    mu1, sigma1 = params1["mu"], params1["sigma"]

    if sigma0 == 0:
        sigma0 = 1

    return np.log(sigma1/sigma0) + (np.square(sigma0) + np.square(mu0 - mu1))/(2 * np.square(sigma1)) - 1/2


def plot_simulation():
    print("Generating plots...")

    observations = [o[0][0]["value"] for o in generated_temps]

    # Plotting the first year of generated and predicted temperatures
    year_time = list(range(YEAR_LEN))
    plot_temperatures(year_time,
                      observations[:YEAR_LEN], "observations",
                      predictions[:YEAR_LEN], "predictions",
                      "First year generated VS predicted temperatures",
                      func="scatter")

    # Plotting the last year of generated and predicted temperatures
    plot_temperatures(year_time,
                      observations[-YEAR_LEN:], "observations",
                      predictions[-YEAR_LEN:], "predictions",
                      "Last year generated VS predicted temperatures",
                      func="scatter")

    plot_states()

    print("Done!")


def plot_temperatures(time, obs, obs_label, pred, pred_label, title, func="scatter"):
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


def plot_states():
    params_per_layer = [l for l in zip(*agent_params)]

    for layer, params in enumerate(params_per_layer):
        final_params = params[-1]

        if len(final_params) < 25:
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
            plt.title("State parameters for layer {} with ±σ as error bars".format(layer))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

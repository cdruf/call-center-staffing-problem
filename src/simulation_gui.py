import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from src.simulation import run_simulation, sample_inter_arrival_time, sample_call_duration_beta, \
    sample_time_until_renege

# %%

# Assumptions
avg_inter_arrival_time = 20
n = 10
conversion_rate = 0.35
avg_profit_per_sale = 200
avg_cost_per_agent_hour = 46.62088498620127
hours = 50
sim_results = run_simulation(n_periods=60 * 60 * hours,
                             n_servers=n,
                             sample_inter_arrival_time_fkt=sample_inter_arrival_time,
                             sample_call_duration_fkt=sample_call_duration_beta,
                             sample_time_until_reneging_fkt=sample_time_until_renege,
                             plot_results=False,
                             avg_renege_time=120,
                             avg_inter_arrival_time=avg_inter_arrival_time)

customers = sim_results.all_customers

# %%
fig = plt.figure()
x = [1, 2, 3]  # 1: being served, 2: waiting, 3: reneged
rects = plt.bar(x, 0,
                color=['c', 'b', 'r'],
                tick_label=["being served", "waiting", "% reneged"])
plt.ylim(0, 100)
plt.grid(True)

customers_waiting = []
customers_in_service = []
n_reneged = 0
n_departed = 0


def animate(t):
    global n_reneged, n_departed

    # Arrivals
    idx = 0
    while idx < len(customers):
        customer = customers[idx]
        if customer.arrival_time <= t:
            customers_waiting.append(customer)
            customers.remove(customer)
        else:
            break  # customers are sorted by arrival time

    # Waiting customers renege or start service
    idx = 0
    while idx < len(customers_waiting):
        customer = customers_waiting[idx]

        # Start service
        if customer.service_start_time <= t:
            customers_in_service.append(customer)
            customers_waiting.remove(customer)

        # Renege
        elif customer.renege_time <= t:
            n_reneged += 1
            customers_waiting.remove(customer)

        else:
            idx += 1

    # End service
    idx = 0
    while idx < len(customers_in_service):
        customer = customers_in_service[idx]
        if customer.departure_time <= t:
            customers_in_service.remove(customer)
            n_departed += 1
        else:
            idx += 1

    rects[0].set_height(len(customers_in_service))
    rects[1].set_height(len(customers_waiting))
    share_reneged = (n_reneged / (n_departed + n_reneged) * 100) if (n_departed + n_reneged > 0) else 0
    rects[2].set_height(share_reneged)
    return rects,


anim = animation.FuncAnimation(fig, animate, frames=sim_results.n_periods,
                               interval=5)
plt.show()

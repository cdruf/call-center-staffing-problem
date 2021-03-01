from dataclasses import dataclass
from datetime import datetime
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy
from scipy.optimize import curve_fit
from scipy.stats import beta

from src import data_folder_path
from src.step_function import StepFunction
from src.util import show_histogram, poly5

# %%

sim_output = False


def print_(text):
    if sim_output:
        print(text)


# %%
"""
# Assumptions
"""


def sample_call_duration_test():
    return np.random.exponential(scale=7.0)


def sample_call_duration_beta():
    a = 1.2396816025688038
    b = 13.148071086271802
    loc = 0.9922421985825189
    scale = 3716.3614369650018
    return beta.rvs(a, b, loc, scale)


def sample_inter_arrival_time(avg_inter_arrival_time, **kwargs):
    return np.random.exponential(scale=avg_inter_arrival_time)


def sample_time_until_reneging_test():
    return np.random.exponential(scale=15.0)


def sample_time_until_renege(avg_renege_time, **kwargs):
    return np.random.exponential(scale=avg_renege_time)


# %%

class Customer:
    next_id = 0
    arrival_times = []
    times_spent_in_queue_till_service = []
    times_spent_in_queue_till_renege = []
    times_spent_in_service = []
    times_spent_in_system = []
    n_renege = 0
    n_served = 0

    @classmethod
    def reset(cls):
        Customer.next_id = 0
        Customer.arrival_times = []
        Customer.times_spent_in_queue_till_service = []
        Customer.times_spent_in_queue_till_renege = []
        Customer.times_spent_in_service = []
        Customer.times_spent_in_system = []
        Customer.n_renege = 0
        Customer.n_served = 0

    def __init__(self, env, queue, arrival_time, sample_time_until_reneging_fkt, **kwargs):
        self.env = env
        self.queue = queue
        self.id = Customer.next_id
        Customer.next_id += 1
        self.arrival_time = arrival_time
        self.sample_time_until_reneging_fkt = sample_time_until_reneging_fkt
        self.service_start_time = np.nan
        self.renege_time = np.nan
        self.departure_time = np.nan
        print_(f'{env.now:.4f}: arrival {self}')

        self.gets_served_event = env.event()
        env.process(self.renege_process())
        self.reneged = False

        self.kwargs = kwargs
        Customer.arrival_times.append(arrival_time)

    def renege_process(self):
        print_(f'{self.env.now:.4f}: {self} start queuing')
        patience = self.sample_time_until_reneging_fkt(**self.kwargs)
        queuing = self.env.timeout(patience)
        ret = yield queuing | self.gets_served_event
        if not self.gets_served_event.triggered:
            self.renege()
        else:
            print_(f"{self.env.now:.4f}: {self} goes to server")

    def renege(self):
        print_(f"{self.env.now:.4f}: {self} reneges after {self.env.now - self.arrival_time:.4f}")
        self.queue.remove(self)
        self.reneged = True
        self.renege_time = self.env.now
        Customer.times_spent_in_queue_till_renege.append(self.env.now - self.arrival_time)
        Customer.times_spent_in_system.append(self.env.now - self.arrival_time)
        Customer.n_renege += 1

    def start_service(self):
        print_(f'{self.env.now:.4f}: start service {self}')
        self.service_start_time = self.env.now
        Customer.times_spent_in_queue_till_service.append(self.env.now - self.arrival_time)

    def departure(self):
        print_(f'{self.env.now:.4f}: departure {self}')
        self.departure_time = self.env.now
        Customer.times_spent_in_service.append(self.env.now - self.service_start_time)
        Customer.times_spent_in_system.append(self.env.now - self.arrival_time)
        Customer.n_served += 1

    def __str__(self):
        return f'Customer {self.id}'


class Server:
    next_id = 0

    def __init__(self, env, sample_call_duration_fkt):
        self.env = env
        self.sample_call_duration_fkt = sample_call_duration_fkt
        self.id = Server.next_id
        Server.next_id += 1
        self.idle = True
        self.serve_start_times = []
        self.serve_end_times = []

    def serve(self, customer, queue):
        assert not customer.reneged
        self.idle = False
        self.serve_start_times.append(self.env.now)
        service_duration = self.sample_call_duration_fkt()
        print_(f'{self.env.now:.4f}: {self} starts serving {customer} for {service_duration:.4f} periods')
        customer.start_service()

        yield self.env.timeout(service_duration)

        customer.departure()
        self.serve_end_times.append(self.env.now)
        self.idle = True

        queue.dispatch()

    def __str__(self):
        return f'Server {self.id}, idle={self.idle}'


class Queue:

    def __init__(self, env, servers: list):
        self.env = env
        self.queue = []
        self.n_customers = 0
        self.n_customers_collector = StepFunction(xs=[0], ys=[])
        self.servers = servers

    def add(self, customer: Customer):
        self.queue.append(customer)
        self.n_customers += 1
        self.n_customers_collector.append_step(self.env.now, self.n_customers)
        print_(f'{self.env.now:.4f}: {customer} entered queue, length {len(self)}')

        self.dispatch()

    def dispatch(self):
        server = self.get_idle_server()
        if self.n_customers > 0 and server is not None:
            customer = self.pop()
            customer.gets_served_event.succeed()
            self.env.process(server.serve(customer, self))

    def get_idle_server(self):
        for server in self.servers:
            if server.idle:
                return server
        return None

    def pop(self) -> Customer:
        customer = self.queue.pop(0)
        self.decrement_customers()
        print_(f'{self.env.now:.4f}: {customer} left queue front, length {len(self)}')
        return customer

    def remove(self, customer):
        assert customer in self.queue
        position = self.queue.index(customer)
        self.queue.remove(customer)
        self.decrement_customers()
        print_(f'{self.env.now:.4f}: {customer} left queue at position {position} of {len(self)}')

    def decrement_customers(self):
        self.n_customers -= 1
        assert self.n_customers >= 0
        self.n_customers_collector.append_step(self.env.now, self.n_customers)

    def __len__(self):
        return self.n_customers

    def __str__(self):
        return f'Queue - length={len(self)}'


def arrival_process(env, queue: Queue, servers: list,
                    sample_inter_arrival_time_fkt,
                    sample_time_until_reneging_fkt,
                    **kwargs):
    while True:
        inter_arrival_time = sample_inter_arrival_time_fkt(**kwargs)
        yield env.timeout(inter_arrival_time)
        customer = Customer(env, queue, env.now,
                            sample_time_until_reneging_fkt,
                            **kwargs)
        queue.add(customer)


# %%
"""
# Run simulation
"""


@dataclass
class SimResult:
    """Class for keeping the simulation results."""
    date_creates: datetime
    n_periods: int
    n_servers: int
    n_arrived: int
    n_served: int
    n_reneged: int
    arrival_times: list
    times_in_queue: list
    times_in_service: list
    personnel_cost: float = 0.0
    revenue: float = 0.0
    profit: float = 0.0
    lost_revenue: float = 0.0


def run_simulation(n_periods=500,
                   n_servers=2,
                   sample_inter_arrival_time_fkt=sample_inter_arrival_time,
                   sample_call_duration_fkt=sample_call_duration_test,
                   sample_time_until_reneging_fkt=sample_time_until_reneging_test,
                   plot_results=True,
                   **kwargs):
    # Reset
    Server.next_id = 0
    Customer.reset()

    # Run
    env = simpy.Environment()
    servers = [Server(env, sample_call_duration_fkt) for i in range(n_servers)]
    queue = Queue(env, servers)
    env.process(arrival_process(env, queue, servers,
                                sample_inter_arrival_time_fkt=sample_inter_arrival_time_fkt,
                                sample_time_until_reneging_fkt=sample_time_until_reneging_fkt,
                                **kwargs))
    env.run(until=n_periods)

    # Results
    print(f"N arrived: {Customer.next_id}")
    print(f"N served: {Customer.n_served}")
    print(f"N reneged: {Customer.n_renege}")
    print(f"N waiting: {len(queue)}")
    n_being_served = reduce(lambda x, y: x + int(not y.idle), servers, 0)
    print(f"N being served: {n_being_served}")

    arrival_times = pd.Series(Customer.arrival_times, dtype=float)
    times_in_queue = pd.Series(Customer.times_spent_in_queue_till_service, dtype=float)
    times_in_service = pd.Series(Customer.times_spent_in_service, dtype=float)

    if plot_results:
        show_histogram(arrival_times, 'Customer arrivals over time')
        show_histogram(times_in_service, 'Time spent in service')
        show_histogram(times_in_queue, 'Time spent in queue', bins=30)
        queue.n_customers_collector.plot("Queue length")

    return SimResult(datetime.now(), n_periods, n_servers,
                     Customer.next_id, Customer.n_served, Customer.n_renege,
                     arrival_times, times_in_queue, times_in_service)


# %%


def run_baseline():
    """
    2020 - 03 - 31
    Capacity: 33 agents, full day with breaks, shifts and breaks at different times
    => ~ 30 agents simultaneously max

    Returns
    -------
    """

    run_simulation(n_periods=60 * 60 * 10,
                   n_servers=22,
                   sample_inter_arrival_time_fkt=sample_inter_arrival_time,
                   sample_call_duration_fkt=sample_call_duration_beta,
                   sample_time_until_reneging_fkt=sample_time_until_renege,
                   avg_renege_time=120,
                   avg_inter_arrival_time=15.517241379310345)


def run_example():
    # Assumptions
    avg_inter_arrival_time = 20
    n = 10
    conversion_rate = 0.35
    avg_profit_per_sale = 200
    avg_cost_per_agent_hour = 46.62088498620127
    hours = 50
    ret = run_simulation(n_periods=60 * 60 * hours,
                         n_servers=n,
                         sample_inter_arrival_time_fkt=sample_inter_arrival_time,
                         sample_call_duration_fkt=sample_call_duration_beta,
                         sample_time_until_reneging_fkt=sample_time_until_renege,
                         plot_results=True,
                         avg_renege_time=120,
                         avg_inter_arrival_time=avg_inter_arrival_time)
    ret.personnel_cost = hours * avg_cost_per_agent_hour * n
    ret.revenue = ret.n_served * conversion_rate * avg_profit_per_sale
    ret.profit = ret.revenue - ret.personnel_cost
    ret.lost_revenue = ret.n_reneged * conversion_rate * avg_profit_per_sale
    print(ret)


def optimize_number_of_agents(avg_inter_arrival_time):
    # Assumptions
    conversion_rate = 0.35
    avg_profit_per_sale = 200
    avg_cost_per_agent_hour = 46.62088498620127
    hours = 50

    # Opt
    ns = []
    profits = []
    maxi = -100000
    argmax = None

    for n in range(10, 100):
        print(f"Run with {n} servers, avg. inter-arrival time = {avg_inter_arrival_time}")
        ns.append(n)
        ret = run_simulation(n_periods=60 * 60 * hours,
                             n_servers=n,
                             sample_inter_arrival_time_fkt=sample_inter_arrival_time,
                             sample_call_duration_fkt=sample_call_duration_beta,
                             sample_time_until_reneging_fkt=sample_time_until_renege,
                             plot_results=False,
                             avg_renege_time=120,
                             avg_inter_arrival_time=avg_inter_arrival_time)
        ret.personnel_cost = hours * avg_cost_per_agent_hour * n
        ret.revenue = ret.n_served * conversion_rate * avg_profit_per_sale
        ret.profit = ret.revenue - ret.personnel_cost
        ret.lost_revenue = ret.n_reneged * conversion_rate * avg_profit_per_sale

        profits.append(ret.profit)

        if ret.profit > maxi:
            maxi = ret.profit
            argmax = ret

    return argmax, ns, profits


def optimize_all():
    data = []
    for avg_arrival_rate in range(5, 60):
        argmax, _, _ = optimize_number_of_agents(avg_arrival_rate)
        row = (avg_arrival_rate, argmax.profit, argmax.n_servers, argmax.n_served, argmax.n_reneged)
        data.append(row)
    df = pd.DataFrame(data, columns=['arrival_rate', 'profit', 'n_servers', 'n_served', 'n_reneged'])
    df.to_csv(data_folder_path / "sim_results.csv", index=False)


# %%

if __name__ == "__main__":
    print('Welcome')
    # run_baseline()
    # run_simulation(n_periods=10000, n_servers=2)
    # ret = optimize_number_of_agents(15)

    # optimize_all()

    # plt.plot(ret[1], ret[2])

    run_example()
    print("Finished")

# %%

import sys

sys.exit(0)

# %%
"""
# Fit a curve to smooth the values
"""

df = pd.read_csv(data_folder_path / "sim_results.csv")
df.head()
x = df['arrival_rate'].to_numpy()
y = df['n_servers'].to_numpy()
plt.plot(x, y)
params, _ = curve_fit(poly5, x, y)
plt.plot(x, poly5(x, *params), color='red')
plt.title("No. servers depending on customer arrival rate")
plt.xlabel('Inter-arrival time (sec)')
plt.ylabel('Number of servers')
plt.tight_layout()
plt.show()

df['n_servers_smoothed'] = poly5(x, *params).round().astype(int)
df.to_csv(data_folder_path / 'sim_results.csv', index=False)

# %%
ret = optimize_number_of_agents(15)
plt.plot(ret[1], ret[2])
plt.title('Profit depending on number of agents with 15 sec inter-arrival time')
plt.xlabel('Number of servers')
plt.ylabel('Profit = revenue - workforce costs')
plt.tight_layout()

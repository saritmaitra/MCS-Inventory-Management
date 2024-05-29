# -*- coding: utf-8 -*-

class Product:
    def __init__(self, name, purchase_cost, lead_time, selling_price, starting_stock, mu, sigma_d, probability, order_cost, holding_cost, expected_demand, sigma_lt, annual_demand):
        self.name = name
        self.purchase_cost = purchase_cost
        self.lead_time = lead_time
        self.selling_price = selling_price
        self.starting_stock = starting_stock
        self.mu = mu
        self.sigma_d = sigma_d  # Demand standard deviation
        self.probability = probability
        self.order_cost = order_cost
        self.holding_cost = holding_cost
        self.expected_demand = expected_demand
        self.sigma_lt = sigma_lt  # Standard deviation over lead time
        self.annual_demand = annual_demand

def calculate_safety_stock(max_daily_sales, max_lead_time, avg_daily_sales, avg_lead_time):
    safety_stock = (max_daily_sales * max_lead_time) - (avg_daily_sales * avg_lead_time)
    return safety_stock

def daily_demand(mean, sd, probability):
    if np.random.uniform(0, 1) > probability:
        return 0
    else:
        return np.exp(np.random.normal(mean, sd))

def monte_carlo_simulation(product, review_period=30, z_score=1.65):
    inventory = 0
    mean = np.log(product.mu)  # Log of mean demand
    sd = product.sigma_d  # Standard deviation of demand
    lead_time = product.lead_time

    daily_sales = [daily_demand(mean, sd, product.probability) for _ in range(365)]
    max_daily_sales = np.max(daily_sales)
    avg_daily_sales = np.mean(daily_sales)

    safety_stock = calculate_safety_stock(max_daily_sales, lead_time, avg_daily_sales, lead_time)

    min_order_qty = 0.2 * avg_daily_sales
    max_order_qty = 0.5 * avg_daily_sales

    q = 0
    stock_out = 0
    counter = 0
    order_placed = False
    data = {'inv_level': [], 'daily_demand': [], 'units_sold': [], 'units_lost': [], 'orders': []}

    for day in range(1, 365):
        day_demand = daily_demand(mean, sd, product.probability)
        data['daily_demand'].append(day_demand)

        if day % review_period == 0:
            q = max(0, safety_stock - inventory)
            q = max(q, min_order_qty)
            q = min(q, max_order_qty)
            order_placed = True
            data['orders'].append(q)

        if order_placed:
            counter += 1

        if counter == lead_time:
            inventory += q
            order_placed = False
            counter = 0

        if inventory - day_demand >= 0:
            data['units_sold'].append(day_demand)
            inventory -= day_demand
        else:
            data['units_sold'].append(inventory)
            data['units_lost'].append(day_demand - inventory)
            inventory = 0
            stock_out += 1

        data['inv_level'].append(inventory)

    return data

def calculate_profit(data, product):
    holding_cost = product.holding_cost
    ordering_cost = product.order_cost
    selling_price = product.selling_price
    days = 365

    revenue = sum(data['units_sold']) * selling_price
    Co = len(data['orders']) * ordering_cost
    Ch = sum(data['inv_level']) * holding_cost / days

    profit = revenue - Co - Ch

    return profit

def run_simulation(product, num_simulations=1000):
    profits = []
    for _ in range(num_simulations):
        data = monte_carlo_simulation(product)
        profit = calculate_profit(data, product)
        profits.append(profit)
    avg_profit = np.mean(profits)
    return avg_profit

def sensitivity_analysis(products, num_simulations=1000):
    for product in products:
        holding_costs = np.linspace(0.1 * product.purchase_cost, 0.5 * product.purchase_cost, 50)
        ordering_costs = np.linspace(500, 1500, 50)
        demand_std_devs = np.linspace(20, 50, 50)
        lead_times = [5, 7, 9, 11, 13, 15, 18, 20]

        profits_holding_costs = []
        profits_ordering_costs = []
        profits_demand_std_devs = []
        profits_lead_times = []

        for holding_cost in holding_costs:
            product.holding_cost = holding_cost
            avg_profit = run_simulation(product, num_simulations)
            profits_holding_costs.append(avg_profit)

        for ordering_cost in ordering_costs:
            product.order_cost = ordering_cost
            avg_profit = run_simulation(product, num_simulations)
            profits_ordering_costs.append(avg_profit)

        for demand_std_dev in demand_std_devs:
            product.sigma_d = demand_std_dev
            avg_profit = run_simulation(product, num_simulations)
            profits_demand_std_devs.append(avg_profit)

        for lead_time in lead_times:
            product.lead_time = lead_time
            avg_profit = run_simulation(product, num_simulations)
            profits_lead_times.append(avg_profit)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.plot(holding_costs, profits_holding_costs, marker='o')
        plt.title(f'Sensitivity Analysis for Holding Costs ({product.name})')
        plt.xlabel('Holding Costs')
        plt.ylabel('Profit')

        plt.subplot(2, 2, 2)
        plt.plot(ordering_costs, profits_ordering_costs, marker='o')
        plt.title(f'Sensitivity Analysis for Ordering Costs ({product.name})')
        plt.xlabel('Ordering Costs')
        plt.ylabel('Profit')

        plt.subplot(2, 2, 3)
        plt.plot(demand_std_devs, profits_demand_std_devs, marker='o')
        plt.title(f'Sensitivity Analysis for Demand Std. Dev. ({product.name})')
        plt.xlabel('Demand Standard Deviation')
        plt.ylabel('Profit')

        plt.subplot(2, 2, 4)
        plt.plot(lead_times, profits_lead_times, marker='o')
        plt.title(f'Sensitivity Analysis for Lead Time ({product.name})')
        plt.xlabel('Lead Time')
        plt.ylabel('Profit')
        plt.tight_layout()
        plt.show()

products = [
    Product(name='PrA', purchase_cost=12, lead_time=9, selling_price=16.10, starting_stock=2750, mu=103.50, sigma_d=37.32, probability=0.76, order_cost=1000, holding_cost=0.2*12, expected_demand=705, sigma_lt=165.01, annual_demand=28670),
    Product(name='PrB', purchase_cost=7, lead_time=6, selling_price=8.60, starting_stock=22500, mu=648.55, sigma_d=26.45, probability=1.00, order_cost=1200, holding_cost=0.2*7, expected_demand=3891, sigma_lt=64.78, annual_demand=237370),
    Product(name='PrC', purchase_cost=6, lead_time=15, selling_price=10.20, starting_stock=5200, mu=201.68, sigma_d=31.08, probability=0.70, order_cost=1000, holding_cost=0.2*6, expected_demand=2266, sigma_lt=383.33, annual_demand=51831),
    Product(name='PrD', purchase_cost=37, lead_time=12, selling_price=68, starting_stock=1400, mu=150.06, sigma_d=3.21, probability=0.23, order_cost=1200, holding_cost=0.2*37, expected_demand=785, sigma_lt=299.92, annual_demand=13056)
]

sensitivity_analysis(products, num_simulations=1000)

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import statsmodels.api as sm

def calculate_safety_stock(max_daily_sales, max_lead_time, avg_daily_sales, avg_lead_time):
    safety_stock = (max_daily_sales * max_lead_time) - (avg_daily_sales * avg_lead_time)
    return safety_stock

summary = {
    'Purchase Cost': [12, 7, 6, 37],
    'Lead Time': [9, 6, 15, 12],
    'Size': [0.57, 0.05, 0.53, 1.05],
    'Selling Price': [16.10, 8.60, 10.20, 68],
    'Starting Stock': [2750, 22500, 5200, 1400],
    'Ch': [0.2 * 12, 0.2 * 7, 0.2 * 6, 0.2 * 37],
    'Co': [1000, 1200, 1000, 1200],
    'Probability': [0.76, 1.00, 0.70, 0.23],
    'Mean Demand (Lead Time)': [103.50, 648.55, 201.68, 150.06],
    'Std. Dev. of Demand (Lead Time)': [37.32, 26.45, 31.08, 3.21],
    'Expected Demand (Lead Time)': [705, 3891, 2266, 785]
}

class Product:
    def __init__(self, i):
        self.mean = np.mean([np.log(j) for j in [summary['Expected Demand (Lead Time)'][i - 1]] if j > 0])
        self.sd = np.std([np.log(j) for j in [summary['Expected Demand (Lead Time)'][i - 1]] if j > 0])
        self.i = i
        self.unit_cost = summary['Purchase Cost'][i - 1]
        self.lead_time = summary['Lead Time'][i - 1]
        self.size = summary['Size'][i - 1]
        self.selling_price = summary['Selling Price'][i - 1]
        self.holding_cost = summary['Ch'][i - 1]
        self.ordering_cost = summary['Co'][i - 1]
        self.probability = summary['Probability'][i - 1]
        self.starting_stock = summary['Starting Stock'][i - 1]
        self.mean_demand_lead_time = summary['Mean Demand (Lead Time)'][i - 1]
        self.std_dev_demand_lead_time = summary['Std. Dev. of Demand (Lead Time)'][i - 1]
        self.expected_demand_lead_time = summary['Expected Demand (Lead Time)'][i - 1]

def daily_demand(mean, sd, probability):
    if np.random.uniform(0, 1) > probability:
        return 0
    else:
        return np.exp(np.random.normal(mean, sd))

def monte_carlo_ray(M, product, review_period=30, z_score=1.65):
    inventory = product.starting_stock
    mean = product.mean
    sd = product.sd
    lead_time = product.lead_time
    probability = product.probability
    demand_lead = product.expected_demand_lead_time  # Corrected attribute name

    # Importance sampling parameters
    high_demand_value = 2 * mean  # Adjust as needed based on the specific scenario
    weights = [0.8, 0.2]  # Adjust weights to emphasize high demand values
    means = [mean, high_demand_value]
    sds = [sd, sd]  # Assuming same standard deviation for simplicity

    # max_daily_sales and avg_daily_sales using the new sampling distribution
    daily_sales = [np.random.normal(means[i], sds[i]) for i in np.random.choice(len(weights), size=365, p=weights)]  # Corrected list comprehension
    max_daily_sales = np.max(daily_sales)
    avg_daily_sales = np.mean(daily_sales)

    # max_lead_time and avg_lead_time
    max_lead_time = lead_time
    avg_lead_time = np.mean(summary['Lead Time'])

    # safety stock using the provided function
    safety_stock = calculate_safety_stock(max_daily_sales, max_lead_time, avg_daily_sales, avg_lead_time)

    # Further calculation remains the same...
    q = 0
    stock_out = 0
    counter = 0
    order_placed = False
    data = {'inv_level': [], 'daily_demand': [], 'units_sold': [], 'units_lost': [], 'orders': [], 'reorder_quantities': []}  # Added 'reorder_quantities'

    for day in range(1, 365):
        # Sample demand from the new distribution
        day_demand = [np.random.normal(means[i], sds[i]) for i in np.random.choice(len(weights), size=365, p=weights)]  # Corrected list comprehension
        if day_demand != 0:  # Add this condition to skip printing when demand is zero
            data['daily_demand'].append(day_demand)

        if day % review_period == 0:
            q = max(0, safety_stock + demand_lead - inventory)
            q = max(q, int(0.2 * mean))  # Minimum order quantity
            order_placed = True
            data['orders'].append(q)
            data['reorder_quantities'].append(q)  # Store reorder quantity

        if order_placed:
            counter += 1

        if counter == lead_time:
            inventory += q
            order_placed = False
            counter = 0

        for demand in day_demand:  # Iterate over each demand value for the current day
            if inventory - demand >= 0:
                data['units_sold'].append(demand)
                inventory -= demand
            else:
                data['units_sold'].append(inventory)
                data['units_lost'].append(demand - inventory)
                inventory = 0
                stock_out += 1

            data['inv_level'].append(inventory)

    return data

def calculate_profit(data, product):
    unit_cost = product.unit_cost
    selling_price = product.selling_price
    holding_cost = product.holding_cost
    order_cost = product.ordering_cost
    size = product.size
    days = 365

    revenue = sum(data['units_sold']) * selling_price
    Co = len(data['orders']) * order_cost
    Ch = sum(data['inv_level']) * holding_cost * size / days
    cost = sum(data['orders']) * unit_cost

    profit = revenue - cost - Co - Ch

    return profit

products = [Product(i) for i in range(1, 5)]
results = {}

for product in products:
    profit_list = []
    reorder_quantity_list = []
    for _ in range(1000):  # Run 1000 simulations for each product
        data = monte_carlo_ray(M=10000, product=product)
        profit = calculate_profit(data, product)
        profit_list.append(profit)
        reorder_quantity_list.extend(data['reorder_quantities'])  # Collect reorder quantities
    results[f'Pr{product.i}'] = {
        'profit_list': profit_list,
        'reorder_quantity_list': reorder_quantity_list
    }

# Plotting convergence checks
for product_name, result in results.items():
    profit_list = result['profit_list']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Running Mean Plot
    running_mean = np.cumsum(profit_list) / np.arange(1, len(profit_list) + 1)
    axes[0].plot(running_mean)
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_ylabel('Running Mean of Profit')
    axes[0].set_title(f'Running Mean Convergence\nfor {product_name}')

    # Batch Means Plot
    batch_size = 100  # Adjust batch size as needed
    num_batches = len(profit_list) // batch_size
    batch_means = [np.mean(profit_list[i * batch_size:(i + 1) * batch_size]) for i in range(num_batches)]
    axes[1].plot(batch_means)
    axes[1].set_xlabel('Batch Number')
    axes[1].set_ylabel('Batch Mean of Profit')
    axes[1].set_title(f'Batch Means Convergence\nfor {product_name}')

    # Standard Error Plot
    standard_errors = [np.std(profit_list[:i]) / np.sqrt(i) for i in range(2, len(profit_list) + 1)]
    axes[2].plot(standard_errors)
    axes[2].set_xlabel('Number of Samples')
    axes[2].set_ylabel('Standard Error of Profit')
    axes[2].set_title(f'Standard Error Convergence\nfor {product_name}')

    # Autocorrelation Plot
    sm.graphics.tsa.plot_acf(profit_list, lags=40, ax=axes[3])
    axes[3].set_xlabel('Lag')
    axes[3].set_ylabel('Autocorrelation')
    axes[3].set_title(f'Autocorrelation of Profit Estimates\nfor {product_name}')

    plt.tight_layout()
    plt.show()
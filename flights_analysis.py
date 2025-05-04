import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import calendar


# Create a report summarizing your insights. Your report should explore the following questions:

# 1. How do different airlines compare in terms of their departure and arrival times? Are there noticeable trends in their on-time performance over the year? A well-structured visualization could help uncover patterns.
# 2. Are there particular months/weeks/time of day where there is a general trend of greater delays in flights across all carriers? If so, what could be the reasons?
# 3. Some airports seem to operate like clockwork, while others are notorious for disruptions. How do different airports compare when it comes to departure and arrival punctuality? Could location, traffic volume, or other factors play a role? Are there patterns that emerge when looking at delays across various airports?
# 4. Predict whether a flight will have a delay of 15 minutes or more at departure.
# 5. What underlying factors influence flight delays the most? Are some routes more prone to disruptions than others? Do external variables like time of day, distance, or carrier policies play a significant role? By analyzing the relationships between different features, you might discover unexpected insights.



# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_path = os.path.join(current_dir, 'flights.csv')

# Load the dataset
df = pd.read_csv(csv_path)
# print(df.head())

# Check the information about the dataset
# print(df.info())

# Check the shape of the dataset (number of rows and columns)
# print(f"Shape of dataset: {df.shape}")

# Check for missing values
# print(df.isnull().sum())

# Check for duplicates
# print(df.duplicated().sum())

# Check the statistics of the dataset
# print(df.describe())

# Drop rows with missing values in critical delay-related columns
df_cleaned = df.dropna(subset=['arr_delay','dep_delay'])

# df_cleaned.to_csv('flights_cleaned.csv', index=False)

# print(df_cleaned.isnull().sum())

# Check the shape of the cleaned dataset
# print(f"Shape of cleaned dataset: {df_cleaned.shape}")

# 1 Airline On-Time Performance

# 1.1 Average Delay per Airline

# Average delay per airline
avg_delay_per_airline = df_cleaned.groupby('name')[['dep_delay', 'arr_delay']].mean().sort_values(by='dep_delay', ascending=False)
# print(avg_delay_per_airline)

# Create a visualization to compare different airlines in terms of their departure and arrival times
# Average departure delay per airline
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_delay_per_airline.reset_index(), x='dep_delay', y='name', palette='viridis')
plt.title('Average Departure Delay per Airline')
plt.xlabel('Average Departure Delay (minutes)')
plt.ylabel('Airline')
plt.tight_layout()
# plt.show()

# Average arrival delay per airline
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_delay_per_airline.reset_index(), x='arr_delay', y='name', palette='winter')
plt.title('Average Arrival Delay per Airline')
plt.xlabel('Average Arrival Delay (minutes)')
plt.ylabel('Airline')
plt.tight_layout()
# plt.show()



# 1.2a Average Delay per Airline by Month

monthly_delay = df_cleaned.groupby(['name', 'month'])[['dep_delay', 'arr_delay']].mean().reset_index()
# print(monthly_delay)

# Find the month with the highest departure delay for each airline
max_dep_delay = monthly_delay.loc[monthly_delay.groupby('name')['dep_delay'].idxmax(), ['name', 'month', 'dep_delay']]
print(f'Month with highest departure delay per airline: {max_dep_delay}') 

# Find the month with the highest arrival delay for each airline
max_arr_delay = monthly_delay.loc[monthly_delay.groupby('name')['arr_delay'].idxmax(), ['name', 'month', 'arr_delay']]
print(f'Month with highest arrival delay per airline: {max_arr_delay}') 

# # # Merge the result for visualization
max_delays = pd.merge(max_dep_delay, max_arr_delay, on='name', suffixes=('_dep', '_arr'))
print(max_delays)

# Visulization of each airline's maximum monthly departure and arrival delays
# Departure Delays with Month Labels
plt.figure(figsize=(14, 6))
sns.barplot(data=max_delays, x='name', y='dep_delay', palette='viridis')
plt.title('Month with Highest Departure Delay for Each Airline')
plt.xlabel('Airline')
plt.ylabel('Avg Departure Delay (minutes)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add month labels on top of the bars
for index, row in max_delays.iterrows():
    month_name = calendar.month_name[int(row['month_dep'])]
    plt.text(index, row['dep_delay'], month_name, color='black', ha="center")
    
plt.show()

# Arrival Delays with Month Labels
plt.figure(figsize=(14, 6))
sns.barplot(data=max_delays, x='name', y='arr_delay', palette='winter')
plt.title('Month with Highest Arrival Delay for Each Airline')
plt.xlabel('Airline')
plt.ylabel('Avg Arrival Delay (minutes)')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels by 45 degrees and align to the right
plt.tight_layout()

# Add month labels on top of the bars
for index, row in max_delays.iterrows():
    month_name = calendar.month_abbr[int(row['month_arr'])]  # Use month abbreviation
    plt.text(index, row['arr_delay'], month_name, color='black', ha="center")

plt.show()

# 1.2b Monthly Delay Trends
# Are there particular months, days, or times when delays tend to increase?
# What patterns emerge across all carriers?

# Monthly average delay
monthly_avg_delay = df_cleaned.groupby('month')[['dep_delay', 'arr_delay']].mean().reset_index()
# print(monthly_avg_delay)

# Create a visualization to show the monthly average delay
plt.figure(figsize=(10,5))
sns.lineplot(data=monthly_avg_delay, x='month', y='dep_delay', marker='o', label='Departure Delay')
sns.lineplot(data=monthly_avg_delay, x='month', y='arr_delay', marker='o', label='Arrival Delay')
plt.title('Monthly Average Delay')
plt.xlabel('Month')
plt.ylabel('Average Delay (minutes)')
plt.xticks(monthly_avg_delay['month'])
plt.legend()
plt.tight_layout()
# plt.show()

# 1.2 Weekly Delay Trends
# Are there particular weeks of the year when delays tend to increase?
# What patterns emerge across all carriers?

# Create the 'date ' column from 'year', 'month', and 'day' columns
df_cleaned['date'] = pd.to_datetime(df_cleaned[['year', 'month', 'day']])

# Extract the week number from the date
df_cleaned['week'] = df_cleaned['date'].dt.isocalendar().week

# Weekly average delay
weekly_avg_delay = df_cleaned.groupby('week')[['dep_delay', 'arr_delay']].mean().reset_index()
# print(weekly_avg_delay)

# Create a visualization to show the weekly average delay
plt.figure(figsize=(10,5))
sns.lineplot(data=weekly_avg_delay, x='week', y='dep_delay', marker='o', color='green',label='Departure Delay')
sns.lineplot(data=weekly_avg_delay, x='week', y='arr_delay', marker='o', color='red',label='Arrival Delay')
plt.title('Weekly Average Delay')
plt.xlabel('Week of the Year')
plt.ylabel('Average Delay (minutes)')
plt.xticks(weekly_avg_delay['week'])
plt.legend()
plt.tight_layout()
# plt.show()

# 1.3 Hourly Delay Trends
# Are there particular hours of the day when delays tend to increase?
# What patterns emerge across all carriers?

# Hourly average delay
hourly_avg_delay = df_cleaned.groupby('hour')[['dep_delay', 'arr_delay']].mean().reset_index()
# print(hourly_avg_delay)

# Create a visualization to show the hourly average delay
plt.figure(figsize=(10,5))
sns.lineplot(data=hourly_avg_delay, x='hour', y='dep_delay', marker='o', label='Departure Delay')
sns.lineplot(data=hourly_avg_delay, x='hour', y='arr_delay', marker='o', label='Arrival Delay')
plt.title('Average Delay by Hour of the Day')
plt.xlabel('Scheduled Departure Hour')
plt.ylabel('Average Delay (minutes)')
plt.xticks(hourly_avg_delay['hour'])
plt.legend()
plt.tight_layout()
# plt.show()


# 2 Airport On-Time Performance
# 2.1 Which airports are the most and least punctual?
# 2.2 Are there regional or traffic-related factors that contribute to these differences?

# 2.1 Average Delay by Origin and Departure Airport
# Top 3 origin airports with highest average departure delays
top_origin_airports = df_cleaned.groupby('origin')[['dep_delay', 'arr_delay']].mean().sort_values(by='dep_delay', ascending=False).head(3)
# print(top_origin_airports)

# Top 10 destination airports with highest average arrival delays
top_dest_airports = df_cleaned.groupby('dest')[['dep_delay', 'arr_delay']].mean().sort_values(by='arr_delay', ascending=False)
print(top_dest_airports.head)


# Create a visualization to show the average departure delay by origin airport
plt.figure(figsize=(10, 5))
sns.barplot(data=top_origin_airports.reset_index().sort_values(by='dep_delay', ascending=False), x='dep_delay', y='origin', palette='rocket')
plt.title('Top 3 Origin Airports with Highest Average Departure Delays')
plt.xlabel('Average Departure Delay (minutes)')
plt.ylabel('Origin Airport')
plt.tight_layout()
# plt.show()

# Create a visualization to show the average arrival delay by destination airport
# Top 10 destination airports with highest average arrival delays
plt.figure(figsize=(10, 5))
sns.barplot(data=top_dest_airports.head(10).reset_index().sort_values(by='arr_delay', ascending=False), x='arr_delay', y='dest', palette='mako')
plt.title('Top 10 Destination Airports with Highest Average Arrival Delays')
plt.xlabel('Average Arrival Delay (minutes)')
plt.ylabel('Destination Airport')
plt.tight_layout()
# plt.show()

# Top 10 destination airports with lowest average arrival delays
plt.figure(figsize=(10,5))
sns.barplot(data=top_dest_airports.tail(10).reset_index().sort_values(by='arr_delay', ascending=False), x='arr_delay', y='dest', palette='rocket')
plt.title('Top 10 Destination Airports with the Lowest Arrival Delays')
plt.xlabel('Average Arrival Delays (minutes)')
plt.ylabel('Destination Airport')
plt.tight_layout()
# plt.show()
import json
import pandas as pd
import numpy as np
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from datetime import datetime, date
from IPython.display import display
import matplotlib.pyplot as plt


with open('transaction-data-adhoc-analysis.json', 'r') as f:
  data = json.load(f)

def convert_quantity(x): ##function to only include number characters https://www.youtube.com/watch?v=-J0ye2LAJ9M
    charset = [*[str(i) for i in range(10)]]
    x = " ".join([i for i in x if i in charset])
    return int(x)

def brand(product):
    if "Candy City" in product:
        return "Candy City"
    elif "Exotic Extras" in product:
        return "Exotic Extras"
    elif "HealthyKid 3+" in product:
        return "HealthyKid 3+"
    
def age(birthdate): #https://www.geeksforgeeks.org/convert-birth-date-to-age-in-pandas/
    today = date.today()
    return today.year - birthdate.year - ((today.month, 
                                      today.day) < (birthdate.month, 
                                                    birthdate.day))
def age_segment(age):
    if age > 0 and age <= 14:
        return "Child (< 14 yo)"
    elif age >= 15 and age <= 24:
        return "Young Adult (15 to 24 yo)"
    elif age >= 25 and age <= 59:
        return "Adult (25 to 59 yo)"
    elif age >= 60:
        return "Senior Citizen (>= 60 yo)"

def addlabels(x,y): #https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        
def spending_type(monthly_exp):
    if monthly_exp < 7500:
        return "Low Monthly Spending Power (< 7,500)"
    elif monthly_exp >=7500 and monthly_exp <=11000:
        return "Average Monthly Spending Power (7,500 to 11,000)"
    elif monthly_exp > 11000 and monthly_exp <= 22000:
        return "Above Average Monthly Spending Power (11,000 to 22,000)"
    elif monthly_exp > 22000:
        return "High Monthly Spending Power (> 22,000)"
    
df = pd.DataFrame(data)

#Cleaning Up the Data
transaction_items_list = df["transaction_items"].tolist()
transaction_ids = np.sort(np.random.randint(low=1e9, high=1e10, size = len(transaction_items_list))) ##creating transaction id
df["transaction_ids"] = np.array(transaction_ids)
df["transaction_items"] = df["transaction_items"].astype("string") ## object to string
df["transaction_items"] = df["transaction_items"].str.split("[;]") ## setting semicolon as split marker
df = df.explode("transaction_items").reset_index(drop=True) ## making new row for different item
df[["product", "quantity"]] = df["transaction_items"].str.split("(", expand = True) ## making a new column for quantity
df["final_quantity"] = df["quantity"].apply(convert_quantity) ## removing ) parenthesis from quantity
df.drop(["quantity", "transaction_items"], inplace=True, axis=1)
df = df[["transaction_ids", "address", "birthdate", "mail", "name", "sex", "username", "product", "final_quantity", "transaction_value", "transaction_date"]]
df.loc[df["product"]=="Exotic Extras,Beef Chicharon,", "product"] = "Exotic Extras, Beef Chicharon" ##adding space after comma and removing last comma for aesthetic purposes
df.loc[df["product"]=="HealthyKid 3+,Nutrional Milk,", "product"] = "HealthyKid 3+, Nutrional Milk"
df.loc[df["product"]=="Candy City,Orange Beans,", "product"] = "Candy City, Orange Beans"
df.loc[df["product"]=="HealthyKid 3+,Gummy Vitamins,", "product"] = "HealthyKid 3+, Gummy Vitamins"
df.loc[df["product"]=="Candy City,Gummy Worms,", "product"] = "Candy City, Gummy Worms"
df.loc[df["product"]=="Exotic Extras,Kimchi and Seaweed,", "product"] = "Exotic Extras, Kimchi and Seaweed"
df.loc[df["product"]=="HealthyKid 3+,Yummy Vegetables,", "product"] = "HealthyKid 3+, Yummy Vegetables"

#Extracting Price
transactions = df[["transaction_ids", "product", "final_quantity", "transaction_value"]]
table = pd.pivot_table(df, values = "final_quantity", index = ["transaction_ids", "transaction_value"], columns = ["product"], fill_value = 0, aggfunc = "sum")
products_table = table.reset_index() ## making dataframe using pivot table to be able to solve the linear equation
products_le = products_table.drop(["transaction_ids", "transaction_value"], axis = 1) ## dropping columns to isolate columns whose values need to be determined
products_arr = np.array(products_le.values.tolist())
totals_arr = np.array(list(products_table["transaction_value"]))
res = lsq_linear(products_arr, totals_arr, bounds=(0, 9999), lsmr_tol='auto', verbose=1) ## getting prices of each product
individual_product_prices = np.array(res.x)
product_names = np.sort(np.array(df["product"].unique()))
price_index = dict(zip(product_names, individual_product_prices)) ## making dictionary with product names as keys
df["product_individual_price"] = df["product"].apply(lambda x: price_index.get(x)) ## adding new column with individual price
df["product_total"] = df['final_quantity'] * df['product_individual_price'] ## adding new column with total of each product 

#Summary Tables
df["transaction_date"] = df["transaction_date"].astype('datetime64[ns]')
df["month"] = df['transaction_date'].dt.strftime('%m/%Y')
summary_table_sales = pd.pivot_table(df, values = ["product_total"], index= "product", columns = "month", aggfunc = np.sum)
summary_table_quantity = pd.pivot_table(df, values = ["final_quantity"], index= "product", columns = "month", aggfunc = np.sum)
display(summary_table_sales)
display(summary_table_quantity)

#Customer Stats

customer_summary = pd.pivot_table(df, values = ["transaction_ids"], index= "username", columns = "month", fill_value = 0, aggfunc= "nunique")
monthly_purchase = customer_summary.reset_index()

#purchasers of each month: greater than 0 purchases
jan_column = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '01/2022')]>0][("username", "")])
feb_column = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '02/2022')]>0][("username", "")])
mar_column = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '03/2022')]>0][("username", "")])
apr_column = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '04/2022')]>0][("username", "")])
may_column = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '05/2022')]>0][("username", "")])
jun_column = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '06/2022')]>0][("username", "")])

#repeaters: shop two consecutive months in a row
jan_repeaters = []
feb_repeaters = np.intersect1d(jan_column, feb_column)
mar_repeaters = np.intersect1d(feb_column, mar_column)
apr_repeaters = np.intersect1d(mar_column, apr_column)
may_repeaters = np.intersect1d(apr_column, may_column)
jun_repeaters = np.intersect1d(may_column, jun_column)

repeaters = [jan_repeaters, feb_repeaters, mar_repeaters, apr_repeaters, may_repeaters, jun_repeaters]
length_checker = np.vectorize(len)
repeaters_count = length_checker(repeaters)

column_values = ["January", "February", "March", "April", "May", "June"]
index_values = ["Repeaters"]

metrics_df = pd.DataFrame(data = repeaters_count, index = column_values, columns = index_values)

#no purchase: 0 purchases
jan_np = np.array([])
feb_np = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '02/2022')]==0][("username", "")])
mar_np = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '03/2022')]==0][("username", "")])
apr_np = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '04/2022')]==0][("username", "")])
may_np = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '05/2022')]==0][("username", "")])
jun_np = np.array(monthly_purchase[monthly_purchase[('transaction_ids', '06/2022')]==0][("username", "")])

#cumulative shoppers for each month
jan_to_feb = np.unique(np.concatenate((jan_column, feb_column), axis = 0))
jan_to_mar = np.unique(np.concatenate((jan_to_feb, mar_column), axis = 0))
jan_to_apr = np.unique(np.concatenate((jan_to_mar, apr_column), axis = 0))
jan_to_may = np.unique(np.concatenate((jan_to_apr, may_column), axis = 0))
jan_to_jun = np.unique(np.concatenate((jan_to_may, jun_column), axis = 0))

#inactive: intersecting shoppers with history but have no purchase for current month
jan_inactive = []
feb_inactive = np.intersect1d(jan_to_feb, feb_np)
mar_inactive = np.intersect1d(jan_to_mar, mar_np)
apr_inactive = np.intersect1d(jan_to_apr, apr_np)
may_inactive = np.intersect1d(jan_to_may, may_np)
jun_inactive = np.intersect1d(jan_to_jun, jun_np)
inactive = [jan_inactive, feb_inactive, mar_inactive, apr_inactive, may_inactive, jun_repeaters]
inactive_count = length_checker(inactive)
metrics_df["Inactive"] = inactive_count

#cumulative streak for each month
jan_engaged = jan_column
feb_engaged = np.intersect1d(jan_column, feb_column)
mar_engaged = np.intersect1d(feb_engaged, mar_column)
apr_engaged = np.intersect1d(mar_engaged, apr_column)
may_engaged = np.intersect1d(apr_engaged, may_column)
jun_engaged = np.intersect1d(may_engaged, jun_column)

engaged = [jan_engaged, feb_engaged, mar_engaged, apr_engaged, may_engaged, jun_engaged]
engaged_count = length_checker(engaged)
metrics_df["Engaged"] = engaged_count
display(metrics_df)

#brand stats

df["brand"] = df["product"].apply(brand)
brand_sales_table = pd.pivot_table(df, values = ["product_total"], index= "brand", columns = "month", aggfunc = np.sum)
brand_quantity_table= pd.pivot_table(df, values = ["final_quantity"], index= "brand", columns = "month", aggfunc = np.sum)
display(brand_sales_table)
display(brand_quantity_table)
#https://www.geeksforgeeks.org/plot-multiple-lines-in-matplotlib/

#qty plot per product
months = ["January", "February", "March", "April", "May", "June"]
qty_plot = round(summary_table_quantity/ 1000, 2)

quantity = qty_plot.to_numpy()
plt.figure(0)
f, ax = plt.subplots(figsize=(12,8))
plt.plot(months, quantity[0], label = "Candy City, Gummy Worms")
plt.plot(months, quantity[1], label = "Candy City, Orange Beans")
plt.plot(months, quantity[2], label = "Exotic Extras, Beef Chicharon")
plt.plot(months, quantity[3], label = "Exotic Extras, Kimchi and Seaweed")
plt.plot(months, quantity[4], label = "HealthyKid 3+, Gummy Vitamins")
plt.plot(months, quantity[5], label = "HealthyKid 3+, Nutrional Milk")
plt.plot(months, quantity[6], label = "HealthyKid 3+, Yummy Vegetables")
plt.xlabel("Month")
plt.ylabel("Units Sold in Thousands")
plt.title("Monthly Unit Growth in Thousands per Product")
plt.legend()
plt.show()

#sales plot per product
sales_plot = round(summary_table_sales/1000000, 2)
sales = sales_plot.to_numpy()
plt.figure(1)
f, ax = plt.subplots(figsize=(12,8))
plt.plot(months, sales[0], label = "Candy City, Gummy Worms")
plt.plot(months, sales[1], label = "Candy City, Orange Beans")
plt.plot(months, sales[2], label = "Exotic Extras, Beef Chicharon")
plt.plot(months, sales[3], label = "Exotic Extras, Kimchi and Seaweed")
plt.plot(months, sales[4], label = "HealthyKid 3+, Gummy Vitamins")
plt.plot(months, sales[5], label = "HealthyKid 3+, Nutrional Milk")
plt.plot(months, sales[6], label = "HealthyKid 3+, Yummy Vegetables")
plt.xlabel("Month")
plt.ylabel("Sales in Millions")
plt.title("Monthly Sales Growth in Millions per Product")
plt.legend()
plt.show()

#total sales line graph
total_sales_table = pd.pivot_table(df, values = ["product_total"], index= "month", aggfunc = np.sum)
total_sales_plot = round(total_sales_table / 1000000, 2)
total_sales = total_sales_table.to_numpy()
plt.figure(2)
f, ax = plt.subplots(figsize = (12,8))
plt.plot(months, total_sales, label = "Total Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales in Millions")
plt.title("Total Sales Growth")
plt.legend()
plt.show()

#total qty line graph
total_qty_table = pd.pivot_table(df, values = ["final_quantity"], index= "month", aggfunc = np.sum)
total_qty_plot = round(total_qty_table/1000, 2)
total_qty = total_qty_plot.to_numpy()
plt.figure(3)
f, ax = plt.subplots(figsize = (12,8))
plt.plot(months, total_qty, label = "Total Units Sold")
plt.xlabel("Month")
plt.ylabel("Total Units in Thousands")
plt.title("Total Unit Growth")
plt.legend()
plt.show()

#pie chart age demographic
ddf = df.groupby(["username", "birthdate"])['transaction_value'].sum().reset_index()
ddf["birthdate"] = ddf["birthdate"].astype('datetime64[ns]')
birthdates = np.array(df["birthdate"])
ddf = ddf.drop_duplicates(subset = ["username"])    
ddf['age'] = ddf['birthdate'].apply(age)
ddf["age_group"]=ddf["age"].apply(age_segment)
ddf_count = ddf.groupby(["age_group"])["age_group"].count().to_frame()
ddf_count
ddf_plot = ddf_count.plot.pie(y='age_group', title = "Customer Age Segmentation", autopct='%1.1f%%', figsize=(8, 8))
ddf_plot.plot()
plt.show()

#agegroup mean total expenditure
ag_qty_table = pd.pivot_table(ddf, values = ["transaction_value"], index= "age_group", aggfunc= np.mean)
ag = ag_qty_table.reset_index()
ag_arr = np.array(ag["age_group"])
mean_plot = np.array(round(ag_qty_table["transaction_value"], 2))
plt.figure(4)
f, ax = plt.subplots(figsize=(12,8))
plt.bar(ag_arr,mean_plot)
plt.xlabel("Age Groups")
plt.ylabel("Mean Total Expenditure")
plt.title("Mean Total Expenditure per Age Group")
addlabels(ag_arr,mean_plot)
plt.show()

#pie chart sex demographic
sdf = pd.DataFrame().assign(username = df["username"], sex = df["sex"])
sdf_count = sdf.groupby(["sex"])["sex"].count().to_frame()
sdf_plot = sdf_count.plot.pie(y="sex", title = "Customer Sex Segmentation", autopct = "%1.1f%%", figsize=(8,8))
sdf_plot.plot()
plt.show()

#best selling products per month
bs = pd.pivot_table(df, values = ["final_quantity"], index = ["product"], columns = ["month"], fill_value = 0, aggfunc = "sum").reset_index()
bs = bs.set_index("product")
bs_df = pd.DataFrame(bs.idxmax())
display(bs_df)

#best selling products per age group
agegroup = dict(zip(ddf.username, ddf.age_group))
df["age_group"] = df["username"].apply(lambda x: agegroup.get(x))
agebs = pd.pivot_table(df, values = "final_quantity", index = "product", columns= "age_group", aggfunc = "sum").reset_index()
agebs = agebs.set_index("product")
agebs_df = pd.DataFrame(agebs.idxmax())
display(agebs_df)

#peaksellingmonth
bs.columns = bs.columns.map('_'.join)
ps = pd.DataFrame(bs.idxmax(axis = 1))
columns = bs.columns
monthd = dict(zip(columns, months))
ps[0] = ps[0].astype(str)
ps["peak_month"] = ps[0].apply(lambda x: monthd.get(x))
ps=ps.drop(ps.columns[0],axis = 1)
display(ps)

#spendingpower piechart and table

pp = pd.pivot_table(df, values = "product_total", index = "username", columns = "month", fill_value = 0, aggfunc = "sum").reset_index()
me = np.array(round(pp.mean(axis = 1), 2))
st = np.vectorize(spending_type)
st_arr = st(me)
pp["mean_monthly_expense"] = me
pp["spending_power"] = st_arr
pp_count = pp.groupby(["spending_power"])["spending_power"].count().to_frame()
display(pp_count)
pp_plot = pp_count.plot.pie(y="spending_power", title = "Spending Power Segmentation", autopct = "%1.1f%%", figsize=(8,8))
pp_plot.plot()
plt.show()


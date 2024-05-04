#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data2 = pd.read_csv('online_shoppers_intention.csv')
data = pd.read_csv('data.csv',sep = ';')


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


selectedArray = pd.Series(data['Administrative_Duration'])
selectedArray2 = pd.Series(data['Informational_Duration'])
selectedArray3 = pd.Series(data['ProductRelated_Duration'])


# In[5]:


standartDev = selectedArray.values.std()


# In[31]:


plt.hist(selectedArray4,)  # alpha, saydamlık
# Grafiği düzenleme
plt.title('Three Histograms')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

# Grafiği göster
plt.show()


# In[33]:


plt.axvline(selectedArray.mean(),linestyle = 'dashed', linewidth = 2,color = 'black')
plt.text(selectedArray.mean(),3,'Mean : {:.2f}'.format(selectedArray.mean()))


# In[ ]:


plt.axvline(selectedArray.mean(), linewidth = 2,color = 'blue')
plt.text(selectedArray.mean(),5,'Median : {:.2f}'.format(selectedArray.mean()))


# In[34]:


plt.axvline(selectedArray.mode().mean(),linestyle = 'dashed', linewidth = 7,color = 'blue')
plt.text(selectedArray.mode().mean(),10,'Mode : {:.2f}'.format(selectedArray.mode().mean()))


# In[4]:


plt.title('Histogram of Rating / Season')
plt.xlabel('Rating')
plt.ylabel('Team Count')

plt.show()


# In[5]:


# Set the size of the figure with a larger height  # Adjust the height value according to your preference

mylabels = ["true", "false"]

plt.pie(selectedArray, labels = mylabels)
plt.legend(title = "Four Fruits:")
plt.show() 
plt.title('Histogram of The weekend or not')
plt.xlabel('Weekend/weekday')
plt.ylabel('Visitors Count')
  # Adjust the range for better x-axis visualization  # Adjust the range for better y-axis (team count) visualization
plt.savefig('weekend.png')


# In[36]:


data1['Administrative']


# In[37]:


print(data2.columns.tolist())


# In[38]:


data2['Administrative']


# In[39]:


data2.rename(columns={0:'Administrative', 1:'Administrative_Duration', 2:'Informational', 3:'Informational_Duration', 4:'ProductRelated', 5:'ProductRelated_Duration', 6:'BounceRates', 7:'ExitRates', 8:'PageValues', 9:'SpecialDay', 10:'Month', 11:'OperatingSystems', 12:'Browser', 13:'Region', 14:'TrafficType', 15:'VisitorType', 16:'Weekend', 17:'Revenue'},inplace=True)


# In[40]:


selectedArray = pd.Series(data2['Administrative'])


# In[41]:


print(data2.columns.tolist())


# In[42]:


fig, ax = plt.subplots()


# In[43]:


ax.pie(selectedArray, radius=1)


# In[44]:


plt.pie(data1['Weekend'])


# In[ ]:


plt.show()


# In[6]:


# Örnek boolean veri seti

site_girisi_series = pd.Series(selectedArray)

# Hafta içi ve hafta sonu günlerini say
hafta_ici = site_girisi_series[site_girisi_series].count()
hafta_sonu = site_girisi_series[~site_girisi_series].count()
# Pie chart grafiği çizimi
labels = ['False', 'True']
sizes = [hafta_ici, hafta_sonu]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # Dilimler arasında boşluk bırakmak için

plt.histplot(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Daireyi daire olarak tutar
plt.title('Pie Chart of the ')

plt.show()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:





# In[7]:


sns.boxplot(data=selectedArray,color='#E99942')
plt.title('The Box and Whisker Plot of Exit Rates')
plt.savefig('exitratesbp.png')
plt.show()


# In[9]:


user_type_counts = data['VisitorType'].value_counts()

# Bar grafiğini çizin
data.plot(kind='bar', color=['#E75185', '#1DBF6C','#3A6DF0'])
plt.title('Returning vs. New Visitors Over Time')
plt.xlabel('User Type')
plt.ylabel('Number of Visitors')
plt.savefig('visitortypesbp.png')
plt.show()


# In[ ]:





# In[10]:


plt.hist([data.SpecialDay,data.Revenue],color=['#0c457d','#0ea7b5'],
         label=["özel gün","dönüş"])
plt.legend()


# In[11]:


data3 = pd.read_csv('weekday.csv',sep=';')


# In[12]:



hafa_ici = pd.Series(data3)

# Hafta içi ve hafta sonu günlerini say
alisveris_yapan = hafta_ici[hafta_ici].count()
alisveris_yapmayan = hafta_ici[~hafta_ici].count()
# Pie chart grafiği çizimi
labels = ['False', 'True']
sizes = [alisveris-yapan, alisveris_yapmayan]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # Dilimler arasında boşluk bırakmak için

plt.histplot(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Daireyi daire olarak tutar
plt.title('Pie Chart of the ')
plt.show()


# In[13]:


selectedArray4 = pd.Series(data3['Revenue'])


# In[14]:


selectedArray4


# In[15]:


df = pd.DataFrame(data3)


# In[16]:


print(df['Revenue'])


# In[17]:


print(df.columns)


# In[14]:


df.columns = df.columns.str.split(';').str[0]

# 'Revenue' sütununu kontrol et
print(df['Revenue'])


# In[18]:


df.columns = df.columns.str.split(';').str[0]


# In[19]:


df.columns


# In[20]:


print(data3)


# In[46]:



hafta_ici_series = pd.Series(selectedArray4)

# Hafta içi ve hafta sonu günlerini say
alisveris_yapan = hafta_ici_series[hafta_ici_series].count()
alisveris_yapmayan = hafta_ici_series[~hafta_ici_series].count()
total= alisveris_yapan + alisveris_yapmayan
# Pie chart grafiği çizimi
labels = ['False', 'True']
sizes = [alisveris_yapmayan, alisveris_yapan]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # Dilimler arasında boşluk bırakmak için

plt.hist(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Daireyi daire olarak tutar
plt.title('Pie Chart of the ')
plt.show()


# In[21]:


hafta_ici_series = pd.Series(selectedArray4)

# Hafta içi ve hafta sonu günlerini say
alisveris_yapan = hafta_ici_series[hafta_ici_series].count()
alisveris_yapmayan = hafta_ici_series[~hafta_ici_series].count()
total= alisveris_yapan + alisveris_yapmayan


# In[22]:


alisveris_yapan


# In[23]:


alisveris_yapmayan


# In[24]:


total


# In[27]:


labels = ['Shoppers', 'Non-shoppers']
percentages = [alisveris_yapan/total, alisveris_yapmayan/total]  # Hafta içi ve hafta sonu yüzdeleri

# Renkleri belirle
colors = ['#3498db', '#e74c3c']

# Bar grafiği çiz
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=percentages, palette=colors)
plt.title('Weekday Visitor Status')
plt.ylabel('Percantage (%)')
plt.show()


# In[28]:


labels = ['Shoppers', 'Non-shoppers']
percentages = [alisveris_yapan, alisveris_yapmayan]  # Hafta içi ve hafta sonu yüzdeleri

# Renkleri belirle
colors = ['#3498db', '#e74c3c']

# Bar grafiği çiz
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=percentages, palette=colors)
plt.title('Weekday Visitor Status')
plt.ylabel('Number of visitor')
plt.show()


# In[30]:


data4 = pd.read_csv('weekend.csv',sep=';')
selectedArray5 = pd.Series(data4['Revenue'])
hafta_sonu_series = pd.Series(selectedArray5)

# Hafta içi ve hafta sonu günlerini say
alisveris_yapan2 = hafta_sonu_series[hafta_sonu_series].count()
alisveris_yapmayan2 = hafta_sonu_series[~hafta_sonu_series].count()
total2= alisveris_yapan2 + alisveris_yapmayan2


# In[54]:


alisveris_yapan2


# In[55]:


alisveris_yapmayan2


# In[56]:


total2


# In[31]:


labels = ['Shoppers', 'Non-shoppers']
percentages = [alisveris_yapan2/total2, alisveris_yapmayan2/total2]  # Hafta içi ve hafta sonu yüzdeleri

# Renkleri belirle
colors = ['#3498db', '#e74c3c']

# Bar grafiği çiz
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=percentages, palette=colors)
plt.title('Weekend Visitor Status')
plt.ylabel('Percentage (%)')
plt.show()


# In[33]:


labels = ['Shoppers', 'Non-shoppers']
percentages = [alisveris_yapan2, alisveris_yapmayan2]  # Hafta içi ve hafta sonu yüzdeleri

# Renkleri belirle
colors = ['#3498db', '#e74c3c']

# Bar grafiği çiz
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=percentages, palette=colors)
plt.title('Weekend Visitor Status')
plt.ylabel('Number of visitor')
plt.show()


# In[61]:


revenue_true_rows = data4[data4['Revenue'] == True]
print(revenue_true_rows)

selectedArray6 = pd.Series(revenue_true_rows['SpecialDay'] )


# In[62]:


selectedArray6


# In[63]:


plt.hist(selectedArray6)


# In[64]:


revenue_true_rows2 = data3[data3['Revenue'] == True]
print(revenue_true_rows2)

selectedArray7 = pd.Series(revenue_true_rows2['SpecialDay'] )


# In[65]:


plt.hist(selectedArray7)


# In[66]:


revenue_true_rows3 = data3[data3['Revenue'] == False]
print(revenue_true_rows3)

selectedArray8 = pd.Series(revenue_true_rows3['SpecialDay'] )


# In[67]:


plt.hist(selectedArray8)


# In[70]:


revenue_true_rows4 = data2[data2['Revenue'] == False]
print(revenue_true_rows4)

selectedArray9 = pd.Series(revenue_true_rows4['SpecialDay'] )


# In[37]:


fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax3.hist(alisveris_yapan,alisveris_yapmayan,alisveris_yapan2,alisveris_yapmayan2, histtype='bar')
ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()


# In[45]:


labels = ['Shoppers weekend', 'Non-shoppers weekend','Shoppers weekday', 'Non-shoppers weekday']
percentages = [alisveris_yapan2/total2, alisveris_yapmayan2/total2,alisveris_yapan/total, alisveris_yapmayan/total]  # Hafta içi ve hafta sonu yüzdeleri

# Renkleri belirle
colors = ['#3498db', '#e74c3c']

# Bar grafiği çiz
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=percentages, palette=colors)
plt.title('Visitor Status')
plt.ylabel('Percentage (%)')
plt.savefig('visitor_status_percantage.png')
plt.show()


# In[46]:


labels = ['Shoppers weekend', 'Non-shoppers weekend','Shoppers weekday', 'Non-shoppers weekday']
percentages = [alisveris_yapan2, alisveris_yapmayan2,alisveris_yapan, alisveris_yapmayan]  # Hafta içi ve hafta sonu yüzdeleri

# Renkleri belirle
colors = ['#3498db', '#e74c3c']

# Bar grafiği çiz
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=percentages, palette=colors)
plt.title('Visitor Status')
plt.ylabel('Number of visitor')
plt.savefig('visitor_status_number.png')
plt.show()


# In[47]:


data7 = pd.read_csv('online_shoppers_intention.csv', sep=';')


# In[85]:


revenue_true_rows8 = data7[data7['Revenue'] == True]
print(revenue_true_rows8)

selectedArray8 = pd.Series(revenue_true_rows8['SpecialDay'] )
revenue_false_rows8 = data7[data7['Revenue'] == False]
print(revenue_true_rows8)
colors = ['#3498db', '#e74c3c']
label =[selectedArray8, selectedArray9]
selectedArray9 = pd.Series(revenue_false_rows8['SpecialDay'] )
plt.hist(selectedArray8,kde= True, bins=10)


# In[80]:


# Verileri filtrele
revenue_true_rows = data7[data7['Revenue'] == True]
revenue_false_rows = data7[data7['Revenue'] == False]

# İki ayrı histogram oluştur
plt.hist([revenue_true_rows['SpecialDay'], revenue_false_rows['SpecialDay']], bins=np.arange(0, 1.1, 0.1), color=['#3498db', '#e74c3c'], label=['True', 'False'], alpha=0.7)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 10500, 1000))
# Eksen etiketleri ve başlık ekle
plt.xlabel('Special Day')
plt.ylabel('Frequency')
plt.title('Histogram of Special Day for Revenue True and False')

# Göster
plt.legend()
plt.show()


# In[43]:


import matplotlib.pyplot as plt
import pandas as pd

# Veri setini yükle
df = pd.read_csv('online_shoppers_intention.csv', sep =';')

# Tarih formatını düzenle
df['Visit_Date'] = pd.to_datetime(df['Month'])

# Tarih aralıklarına göre grupla
daily_sales = df.groupby(df['Month'].dt.date)['Purchase'].mean()

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 5))
plt.plot(daily_sales.index, daily_sales.values, marker='o')
plt.title('Günlük Alışveriş Oranları')
plt.xlabel('Tarih')
plt.ylabel('Ortalama Satın Alma Oranı')
plt.show()


# In[21]:


month_mapping = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'June': '06',
                 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

df['Visit_Date'] = df['Month'].map(month_mapping)

# Tam tarih formatına dönüştürme (örneğin, 2023-02 gibi)
df['Visit_Date'] = '2023-' + df['Visit_Date']

# Tarih sütununu datetime formatına çevirme
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'], format='%Y-%m')

# Sonuçları görüntüleme
print(df)


# In[22]:


import pandas as pd

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Purchase'].mean()

# Çizgi grafiği ile görselleştirme
monthly_sales.plot(kind='line', marker='o')
plt.title('Aylık Ortalama Satın Alma Oranları')
plt.xlabel('Ay')
plt.ylabel('Ortalama Satın Alma Oranı')
plt.show()


# In[45]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükle

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Çizgi grafiği ile görselleştirme
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.savefig('month_analysis.png')
plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset

# Format the 'Month' column and assign it to the 'Visit_Date' column
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Check for invalid values (optional)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Group by date intervals
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Adjust the order of months
months_order = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

# Visualize with a line plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rates')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')

# Set x-axis labels in chronological order
plt.xticks(monthly_sales.index.strftime('%b %y').unique(), rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset

# Define a function to convert month abbreviations to numerical values
def month_to_number(month):
    return {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }[month]

# Convert 'Month' column to numerical values
df['Month_Num'] = df['Month'].apply(month_to_number)

# Sort the DataFrame based on the numerical month values
df = df.sort_values(by='Month_Num')

# Visualize with a line plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(df['Month'], df['Revenue'], marker='o')
plt.title('Monthly Average Purchase Rates')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset

# Define a function to convert full month names to numerical values
def month_to_number(month):
    return {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }[month[:3]]  # Use the first three characters for comparison

# Convert 'Month' column to numerical values
df['Month_Num'] = df['Month'].apply(month_to_number)

# Sort the DataFrame based on the numerical month values
df = df.sort_values(by='Month_Num')

# Visualize with a line plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(df['Month'], df['Revenue'], marker='o')
plt.title('Monthly Average Purchase Rates')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükle

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[26]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükle

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Tüm ayları içerecek şekilde indeksleme
all_months = pd.period_range(min(monthly_sales.index), max(monthly_sales.index), freq='M')
monthly_sales = monthly_sales.reindex(all_months)

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Veri setini yükle

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Ayların sıralı bir şekilde listesi
ordered_months = [calendar.month_abbr[i] for i in range(1, 13)]

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.loc[ordered_months].plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[28]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükle

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Ayların sıralı bir şekilde listesi
ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(range(len(ordered_months)), ordered_months, rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükle

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Ayların sıralı bir şekilde listesi
ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(range(len(ordered_months)), ordered_months, rotation=45, ha='right')

# X ekseni sınırlarını ayarla
plt.xlim(-0.5, len(ordered_months) - 0.5)

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükle

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

# Ayların sıralı bir şekilde listesi
ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(range(len(ordered_months)), ordered_months, rotation=45, ha='right')

# X ekseni sınırlarını ayarla
plt.xlim(-0.5, len(ordered_months) - 0.5)

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt

# 'Month' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Month'] + ' 01', format='%b %y', errors='coerce')

# Hatalı değerleri kontrol et (opsiyonel)
invalid_dates = df[df['Visit_Date'].isnull()]
print("Invalid Dates:")
print(invalid_dates[['Month', 'Visit_Date']])

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Revenue'].mean()

print("Monthly Sales:")
print(monthly_sales)

# Ayların sıralı bir şekilde listesi
ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(range(len(ordered_months)), ordered_months, rotation=45, ha='right')

# X ekseni sınırlarını ayarla
plt.xlim(-0.5, len(ordered_months) - 0.5)

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[33]:


# Ayların sıralı bir şekilde listesi
ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')

# X ekseni etiketlerini ve sınırlarını ayarla
plt.xticks(range(len(ordered_months)), ordered_months, rotation=45, ha='right')
plt.xlim(-0.5, len(ordered_months) - 0.5)

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[34]:


# Ayların sıralı bir şekilde listesi
ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Monthly Sales verileri
monthly_sales_data = {
    '2001-02': 0.016304,
    '2001-03': 0.100682,
    '2001-05': 0.108502,
    '2001-07': 0.152778,
    '2001-08': 0.175520,
    '2001-09': 0.191964,
    '2001-10': 0.209472,
    '2001-11': 0.253502,
    '2001-12': 0.125072
}

# Eksik ayları tamamla
filled_monthly_sales_data = {month: monthly_sales_data.get(month, 0) for month in ordered_months}

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(ordered_months, list(filled_monthly_sales_data.values()), marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Ensure proper layout
plt.show()


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt

# Verileri oluştur
data = {
    'Visit_Date': ['2001','2001-06', '2001-03', '2001-05', '2001-07', '2001-08', '2001-09', '2001-10', '2001-11', '2001-12'],
    'Monthly_Sales': [0.016304, 0.100682, 0.108502, 0.152778, 0.175520, 0.191964, 0.209472, 0.253502, 0.125072]
}

# Veri çerçevesini oluştur
df = pd.DataFrame(data)

# 'Visit_Date' sütununu düzenle ve 'Visit_Date' sütununa at
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'] + ' 01', format='%Y-%m %d', errors='coerce')

# Tarih aralıklarına göre grupla
monthly_sales = df.groupby(df['Visit_Date'].dt.to_period("M"))['Monthly_Sales'].mean()

# Çizgi grafiği ile görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o')
plt.title('Monthly Average Purchase Rate')
plt.xlabel('Month')
plt.ylabel('Average Purchase Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# In[46]:


import seaborn as sns

# Sayfa türlerini grupla ve toplam alışveriş sayısını hesapla
page_types = ['Product', 'Informational', 'Administrative']
page_grouped = df.groupby('Page_Type')[['Revenue']].sum().reindex(page_types)

# Bar grafiği ile görselleştirme
plt.figure(figsize=(8, 5))
sns.barplot(x=page_grouped.index, y='Purchase', data=page_grouped)
plt.title('Sayfa Türlerine Göre Toplam Satın Alma Sayısı')
plt.xlabel('Sayfa Türleri')
plt.ylabel('Toplam Satın Alma')
plt.show()


# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sayfa türlerini grupla
page_types = ['Administrative', 'Informational', 'ProductRelated']
page_grouped = df.groupby('Page_Type')['Purchase'].mean().reindex(page_types)

# Veriyi ısı haritasına dönüştür
heatmap_data = df.groupby(['Page_Type', 'Purchase']).size().unstack(fill_value=0)

# Normalize et
heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

# Heatmap oluştur
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2%", cbar_kws={'label': 'Alışveriş Yüzdesi'})
plt.title('Sayfa Türlerine Göre Alışveriş Yüzdesi')
plt.xlabel('Alışveriş Durumu')
plt.ylabel('Sayfa Türleri')
plt.show()


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sayfa türlerini grupla ve ortalama süreleri hesapla
page_types = ['Administrative', 'Informational', 'ProductRelated']
for page_type in page_types:
    avg_duration = df[df[page_type] > 0]['Duration'].mean()
    print(f"Ortalama süre {page_type}: {avg_duration:.2f} saniye")

# Boxplot ile görselleştirme
plt.figure(figsize=(10, 6))
sns.boxplot(x="Page_Type", y="Duration", data=df, order=page_types)
plt.title('Sayfa Türlerine Göre Ziyaretçi Süreleri')
plt.xlabel('Sayfa Türleri')
plt.ylabel('Ziyaretçi Süreleri (saniye)')
plt.show()


# In[50]:


# Visitor Type ile alışveriş ilişkisi
plt.figure(figsize=(8, 5))
sns.countplot(x='VisitorType', hue='Revenue', data=df)
plt.title('Ziyaretçi Türüne Göre Alışveriş İlişkisi')
plt.xlabel('Visitor Type')
plt.ylabel('Count')
plt.show()

# Weekend ile alışveriş ilişkisi
plt.figure(figsize=(8, 5))
sns.countplot(x='Weekend', hue='Revenue', data=df)
plt.title('Hafta İçi/Hafta Sonu Alışveriş İlişkisi')
plt.xlabel('Weekend')
plt.ylabel('Count')
plt.show()

# Ay ile alışveriş ilişkisi
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', hue='Revenue', data=df)
plt.title('Ay İle Alışveriş İlişkisi')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()



# In[55]:


plt.figure(figsize=(8, 5))
sns.histplot(x='Informational_Duration', hue='Revenue', data=df, kde=True)
plt.title('Informational Duration İle Alışveriş İlişkisi')
plt.xlabel('Informational Duration (seconds)')
plt.ylabel('Count')
plt.show()


# In[57]:


plt.figure(figsize=(8, 5))
sns.histplot(x='Informational_Duration', hue='Revenue', data=df, kde=True)
plt.title('Informational Duration İle Alışveriş İlişkisi')
plt.xlabel('Informational Duration (seconds)')
plt.ylabel('Count')
plt.show()


# In[59]:


# Relationship between Visitor Type and Purchase
plt.figure(figsize=(8, 5))
sns.countplot(x='VisitorType', hue='Revenue', data=df)
plt.title('Purchase Relationship by Visitor Type')
plt.xlabel('Visitor Type')
plt.ylabel('Count')
plt.show()

# Relationship between Weekend and Purchase
plt.figure(figsize=(8, 5))
sns.countplot(x='Weekend', hue='Revenue', data=df)
plt.title('Purchase Relationship by Weekend')
plt.xlabel('Weekend')
plt.ylabel('Count')
plt.show()

# Relationship between Month and Purchase
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', hue='Revenue', data=df)
plt.title('Purchase Relationship by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()


# In[65]:


# Visitor Type ile alışveriş ilişkisi yüzdelik olarak
plt.figure(figsize=(8, 5))
visitor_type_purchase = df.groupby(['VisitorType', 'Revenue']).size().unstack()
visitor_type_purchase_percentage = visitor_type_purchase.div(visitor_type_purchase.sum(axis=1), axis=0) * 100
visitor_type_purchase_percentage.plot(kind='bar', stacked=True)
plt.title('Purchase Relationship by Visitor Type (Percentage)')
plt.xlabel('Visitor Type')
plt.savefig('vtyuzde.png')
plt.ylabel('Percentage')
plt.show()

# Weekend ile alışveriş ilişkisi yüzdelik olarak
plt.figure(figsize=(8, 5))
weekend_purchase = df.groupby(['Weekend', 'Revenue']).size().unstack()
weekend_purchase_percentage = weekend_purchase.div(weekend_purchase.sum(axis=1), axis=0) * 100
weekend_purchase_percentage.plot(kind='bar', stacked=True)
plt.title('Purchase Relationship by Weekend (Percentage)')
plt.xlabel('Weekend')
plt.savefig('wyuzde.png')
plt.ylabel('Percentage')
plt.show()

# Ay ile alışveriş ilişkisi yüzdelik olarak
plt.figure(figsize=(12, 6))
month_purchase = df.groupby(['Month', 'Revenue']).size().unstack()
month_purchase_percentage = month_purchase.div(month_purchase.sum(axis=1), axis=0) * 100
month_purchase_percentage.plot(kind='bar', stacked=True)
plt.title('Purchase Relationship by Month (Percentage)')
plt.xlabel('Month')
plt.ylabel('mPercentage')
plt.savefig('yuzde.png')
plt.show()


# In[69]:


plt.figure(figsize=(80, 50))
sns.histplot(x='Informational_Duration', hue='Revenue', data=df, kde=True)
plt.title('Informational Duration İle Alışveriş İlişkisi')
plt.xlabel('Informational Duration (seconds)')
plt.ylabel('Count')
plt.show()


# In[67]:


plt.figure(figsize=(8, 5))
sns.histplot(x='ProductRelated_Duration', hue='Revenue', data=df, kde=True)
plt.title('ProductRelated Duration İle Alışveriş İlişkisi')
plt.xlabel('ProductRelated Duration (seconds)')
plt.ylabel('Count')
plt.ylim(0, 20)  # Y eksenini 0 ile 100 arasında sınırla
plt.show()


# In[68]:


plt.figure(figsize=(8, 5))
sns.histplot(x='ProductRelated_Duration', hue='Revenue', data=df, kde=True)
plt.title('ProductRelated Duration İle Alışveriş İlişkisi')
plt.xlabel('ProductRelated Duration (seconds)')
plt.ylabel('Count')
plt.ylim(0, 10)  # Y eksenini 0 ile 100 arasında sınırla
plt.show()


# In[72]:


plt.figure(figsize=(8, 5))
sns.histplot(x='ProductRelated_Duration', hue='Revenue', data=df, kde=True, bins=range(0, 3000, 100))
plt.title('ProductRelated Duration İle Alışveriş İlişkisi')
plt.xlabel('ProductRelated Duration (seconds)')
plt.ylabel('Count')
plt.xlim(0,3000,100)
plt.ylim(0, 10)  # Y eksenini 0 ile 100 arasında sınırla
plt.show()


# In[75]:


plt.figure(figsize=(8, 5))
sns.histplot(x='ProductRelated_Duration', hue='Revenue', data=df, kde=True, bins=range(0, 3000, 100))
plt.title('ProductRelated Duration İle Alışveriş İlişkisi')
plt.xlabel('ProductRelated Duration (seconds)')
plt.ylabel('Count')
plt.xticks(range(0, 3000, 1000))  # X ekseni etiketlerini 100'ün katları olarak göster
plt.ylim(0, 100)  # Y eksenini 0 ile 100 arasında sınırla
plt.show()


# In[76]:


plt.figure(figsize=(8, 5))
sns.boxplot(x='Revenue', y='ProductRelated_Duration', data=df)
plt.title('ProductRelated Duration İle Alışveriş İlişkisi')
plt.xlabel('Revenue')
plt.ylabel('ProductRelated Duration (seconds)')
plt.show()


# In[77]:


print(df['ProductRelated_Duration'].describe())
print(df['ProductRelated_Duration'].isnull().sum())


# In[78]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 'ProductRelated_Duration' sütununu sayısal bir formata çevir
df['ProductRelated_Duration'] = pd.to_numeric(df['ProductRelated_Duration'], errors='coerce')

# NaN değerlere sahip satırları düşür
df = df.dropna(subset=['ProductRelated_Duration'])

# Boxplot çizimi
plt.figure(figsize=(8, 5))
sns.boxplot(x='Revenue', y='ProductRelated_Duration', data=df)
plt.title('ProductRelated Duration İle Alışveriş İlişkisi')
plt.xlabel('Revenue')
plt.ylabel('ProductRelated Duration (seconds)')
plt.show()


# In[79]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 'Administrative_Duration' sütununu sayısal bir formata çevir
df['Administrative_Duration'] = pd.to_numeric(df['Administrative_Duration'], errors='coerce')

# NaN değerlere sahip satırları düşür
df_admin = df.dropna(subset=['Administrative_Duration'])

# Boxplot çizimi
plt.figure(figsize=(8, 5))
sns.boxplot(x='Revenue', y='Administrative_Duration', data=df_admin)
plt.title('Administrative Duration İle Alışveriş İlişkisi')
plt.xlabel('Revenue')
plt.ylabel('Administrative Duration (seconds)')
plt.show()

# 'Informational_Duration' sütununu sayısal bir formata çevir
df['Informational_Duration'] = pd.to_numeric(df['Informational_Duration'], errors='coerce')

# NaN değerlere sahip satırları düşür
df_info = df.dropna(subset=['Informational_Duration'])

# Boxplot çizimi
plt.figure(figsize=(8, 5))
sns.boxplot(x='Revenue', y='Informational_Duration', data=df_info)
plt.title('Informational Duration İle Alışveriş İlişkisi')
plt.xlabel('Revenue')
plt.ylabel('Informational Duration (seconds)')
plt.show()


# In[80]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 'Administrative_Duration' sütununu sayısal bir formata çevir
df_copy = df.copy()  # Orijinal DataFrame'i değiştirmemek için bir kopya oluştur
df_copy['Administrative_Duration'] = pd.to_numeric(df_copy['Administrative_Duration'], errors='coerce')

# NaN değerlere sahip satırları düşür
df_admin = df_copy.dropna(subset=['Administrative_Duration'])

# Boxplot çizimi
plt.figure(figsize=(8, 5))
sns.boxplot(x='Revenue', y='Administrative_Duration', data=df_admin)
plt.title('Administrative Duration İle Alışveriş İlişkisi')
plt.xlabel('Revenue')
plt.ylabel('Administrative Duration (seconds)')
plt.show()


# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sayısal sütunları seç
numeric_columns = df.select_dtypes(include=['number'])

# Korelasyon matrisini oluştur
correlation_matrix = numeric_columns.corr()

# Isı haritasını çiz
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Korelasyon Matrisi Heatmap')
plt.show()


# In[82]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 'Revenue' sütununu binary formata çevir
df['Revenue_Binary'] = df['Revenue'].map({True: 1, False: 0})

# Sayısal sütunları seç
numeric_columns = df.select_dtypes(include=['number'])

# Korelasyon matrisini oluştur
correlation_matrix = numeric_columns.corr()

# 'Revenue_Binary' sütununu ekleyerek Isı haritasını çiz
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Korelasyon Matrisi Heatmap (with Revenue_Binary)')
plt.show()


# In[84]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 'Revenue' sütununu binary formata çevir
df['Revenue_Binary'] = df['Revenue'].map({True: 1, False: 0})

# Sayısal sütunları seç
numeric_columns = df.select_dtypes(include=['number'])

# Korelasyon matrisini oluştur
correlation_matrix = numeric_columns.corr()

# 'Revenue_Binary' sütununu ekleyerek Isı haritasını çiz
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Korelasyon Matrisi Heatmap (with Revenue_Binary)')
plt.show()


# In[88]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 'Revenue' sütununu binary formata çevir
df['Revenue_Binary'] = df['Revenue'].map({True: 1, False: 0})

# 'TrafficType', 'Region' ve 'Browser' sütunlarını çıkart
columns_to_exclude = ['TrafficType', 'Region', 'Browser', 'OperatingSystems']
numeric_columns = df.select_dtypes(include=['number']).drop(columns=columns_to_exclude)

# Korelasyon matrisini oluştur
correlation_matrix = numeric_columns.corr()

# 'Revenue_Binary' sütununu ekleyerek Isı haritasını çiz
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation.png')
plt.show()


# In[105]:


import pandas as pd
from scipy.stats import kendalltau
df1 = pd.read_csv('online_shoppers_intention.csv', sep=';')

df1['Revenue'] = df1['Revenue'].map({True: 1, False: 0})
df1['VisitorType'] = df1['VisitorType'].map({'New_Visitor': 0, 'Returning_Visitor': 1})

# Kendall'ın Tau-b korelasyonunu hesapla
corr, _ = kendalltau(df1['Revenue'], df1['VisitorType'])

print(f"Kendall's Tau-b Korelasyon Katsayısı: {corr}")


# In[99]:



# String değerleri binary (0 ve 1) değerlere dönüştür
df['Revenue'] = df['Revenue'].map({'True': 1, 'False': 0}).copy()
df['VisitorType'] = df['VisitorType'].map({'New_Visitor': 0, 'Returning_Visitor': 1}).copy()

# Kendall'ın Tau-b korelasyonunu hesapla
corr, _ = kendalltau(df['Revenue'], df['VisitorType'])

print(f"Kendall's Tau-b Korelasyon Katsayısı: {corr}")


# In[107]:


from scipy.stats import kendalltau

# Boş değerleri içeren satırları düşür
df1_cleaned = df1[['Revenue', 'VisitorType']].dropna()

# Kendall'ın Tau-b korelasyonunu hesapla
corr, _ = kendalltau(df1_cleaned['Revenue'], df1_cleaned['VisitorType'])

print(f"Kendall's Tau-b Korelasyon Katsayısı: {corr}")


# In[108]:


print("Revenue Değerleri ve Frekansları:")
print(df1_cleaned['Revenue'].value_counts())

print("\nVisitorType Değerleri ve Frekansları:")
print(df1_cleaned['VisitorType'].value_counts())


# In[109]:


import matplotlib.pyplot as plt
import seaborn as sns

# 'Revenue' ve 'VisitorType' sütunlarının frekans tablosunu oluştur
revenue_freq = df1_cleaned['Revenue'].value_counts()
visitor_type_freq = df1_cleaned['VisitorType'].value_counts()

# Subplot oluştur
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Revenue frekansları için bar plot
sns.barplot(x=revenue_freq.index, y=revenue_freq.values, ax=axes[0])
axes[0].set_title('Revenue Frekansları')
axes[0].set_xlabel('Revenue')
axes[0].set_ylabel('Frekans')

# VisitorType frekansları için bar plot
sns.barplot(x=visitor_type_freq.index, y=visitor_type_freq.values, ax=axes[1])
axes[1].set_title('VisitorType Frekansları')
axes[1].set_xlabel('VisitorType')
axes[1].set_ylabel('Frekans')

plt.tight_layout()
plt.show()


# In[111]:


# 0 ve 1'ı temsil eden etiketleri oluştur
revenue_labels = {0: 'False', 1: 'True'}
visitor_type_labels = {0: 'New Visitor', 1: 'Returning Visitor'}

# 'Revenue' ve 'VisitorType' sütunlarına etiketleri ekle
df1_cleaned['Revenue_Label'] = df1_cleaned['Revenue'].map(revenue_labels)
df1_cleaned['VisitorType_Label'] = df1_cleaned['VisitorType'].map(visitor_type_labels)

# Frekans tablosunu tekrar oluştur
revenue_freq = df1_cleaned['Revenue_Label'].value_counts()
visitor_type_freq = df1_cleaned['VisitorType_Label'].value_counts()

# Subplot oluştur
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Revenue frekansları için bar plot
sns.barplot(x=revenue_freq.index, y=revenue_freq.values, ax=axes[0])
axes[0].set_title('Revenue Frekansları')
axes[0].set_xlabel('Revenue')
axes[0].set_ylabel('Frekans')

# VisitorType frekansları için bar plot
sns.barplot(x=visitor_type_freq.index, y=visitor_type_freq.values, ax=axes[1])
axes[1].set_title('VisitorType Frekansları')
axes[1].set_xlabel('VisitorType')
axes[1].set_ylabel('Frekans')

plt.tight_layout()
plt.show()


# In[112]:


# Create labels representing what 0 and 1 signify in the data
revenue_labels = {0: 'No Revenue', 1: 'Revenue'}
visitor_type_labels = {0: 'New Visitor', 1: 'Returning Visitor'}

# Add labels to the 'Revenue' and 'VisitorType' columns
df1_cleaned['Revenue_Label'] = df1_cleaned['Revenue'].map(revenue_labels)
df1_cleaned['VisitorType_Label'] = df1_cleaned['VisitorType'].map(visitor_type_labels)

# Recreate the frequency tables
revenue_freq = df1_cleaned['Revenue_Label'].value_counts()
visitor_type_freq = df1_cleaned['VisitorType_Label'].value_counts()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot for revenue frequencies
sns.barplot(x=revenue_freq.index, y=revenue_freq.values, ax=axes[0])
axes[0].set_title('Revenue Frequencies')
axes[0].set_xlabel('Revenue')
axes[0].set_ylabel('Frequency')

# Bar plot for VisitorType frequencies
sns.barplot(x=visitor_type_freq.index, y=visitor_type_freq.values, ax=axes[1])
axes[1].set_title('VisitorType Frequencies')
axes[1].set_xlabel('VisitorType')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[113]:


from scipy.stats import kendalltau

# Drop rows containing missing values
df1_cleaned = df1[['Revenue', 'VisitorType']].dropna()

# Calculate Kendall's Tau-b correlation
corr, _ = kendalltau(df1_cleaned['Revenue'], df1_cleaned['VisitorType'])

print(f"Kendall's Tau-b Correlation Coefficient: {corr}")


# In[ ]:





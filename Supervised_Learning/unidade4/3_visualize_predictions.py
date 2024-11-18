
from src.utils import processing_sales_clean
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

X,y,predictions = processing_sales_clean()

# Create scatter plot
plt.scatter(X, y, color='blue')

# Create line plot
plt.plot(X, y, color="yellow")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()
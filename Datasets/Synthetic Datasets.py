import pandas as pd
import numpy as np

np.random.seed(0)

cluster1 = np.random.normal(loc=[-9, -1], scale=0.5, size=(100, 2))
cluster2 = np.random.normal(loc=[-7, -6], scale=0.6, size=(100, 2))
cluster3 = np.random.normal(loc=[-1, -7], scale=0.6, size=(100, 2))
data = np.concatenate([cluster1, cluster2, cluster3])
df1 = pd.DataFrame(data, columns=['x', 'y'])
df1['real labels'] = np.repeat([0, 1, 2], 100)


cluster1 = np.random.normal(loc=[-2, 3], scale=0.5, size=(100, 2))
cluster2 = np.random.normal(loc=[-2, -3], scale=0.5, size=(100, 2))
cluster3 = np.random.normal(loc=[-8, -4], scale=0.5, size=(100, 2))
data = np.concatenate([cluster1, cluster2, cluster3])
df2 = pd.DataFrame(data, columns=['x', 'y'])
df2['real labels'] = np.repeat([0, 1, 2], 100)


cluster1 = np.random.normal(loc=[-6, 2], scale=0.7, size=(100, 2))
cluster2 = np.random.normal(loc=[-9, -5], scale=0.7, size=(100, 2))
cluster3 = np.random.normal(loc=[-10, -1.5], scale=0.7, size=(100, 2))
data3 = np.concatenate([cluster1, cluster2, cluster3])
df3 = pd.DataFrame(data3, columns=['x', 'y'])
df3['real labels'] = np.repeat([0, 1, 2], 100)


theta1 = np.random.uniform(-np.pi, 0, 150)
r1 = 0.5 + 0.05 * np.random.randn(150)
x1 = r1 * np.cos(theta1)
y1 = 0.5 + r1 * np.sin(theta1)
theta2 = np.random.uniform(0, np.pi, 150)
r2 = 0.5 + 0.05 * np.random.randn(150)
x2 = 0.5 + r2 * np.cos(theta2)
y2 = 0.5 + r2 * np.sin(theta2)
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
df4 = pd.DataFrame({'x': x, 'y': y})
df4['real labels'] = np.repeat([0, 1], 150)


t = np.linspace(0, 2*np.pi, 200)
r_values = [0.3, 0.6, 0.9] 
data5 = []
for r in r_values:
    x = r * np.cos(t) + np.random.normal(scale=0.025, size=t.shape)
    y = r * np.sin(t) + np.random.normal(scale=0.025, size=t.shape)
    data5.append(np.vstack([x, y]).T)
df5 = pd.DataFrame(np.concatenate(data5), columns=['x', 'y'])
df5['real labels'] = np.repeat([0, 1, 2], 200)


x = np.linspace(-0.5, 1.5, 200) 
y1 = -x + np.random.normal(scale=0.05, size=x.shape) 
y2 = -x + 0.5 + np.random.normal(scale=0.05, size=x.shape) 
y3 = -x + 1.0 + np.random.normal(scale=0.05, size=x.shape) 
data6 = np.concatenate([np.vstack([x, y1]).T, np.vstack([x, y2]).T, np.vstack([x, y3]).T])
df6 = pd.DataFrame(data6, columns=['x', 'y'])
df6['real labels'] = np.repeat([0, 1, 2], 200)


x1 = np.full(100, 2) + 0.1 * np.random.randn(100)
y1 = np.random.uniform(2, 4.5, 100)
x2 = np.random.uniform(2, 3.5, 100)
y2 = np.full(100, 2) + 0.1 * np.random.randn(100)
theta1 = np.random.uniform(-np.pi, 0, 100)
r1 = 0.5 + 0.05 * np.random.randn(100)
x3 = 1 + r1 * np.cos(theta1)
y3 = 3 + r1 * np.sin(theta1)
theta2 = np.random.uniform(0, np.pi, 100)
r2 = 0.5 + 0.05 * np.random.randn(100)
x4 = 3.5 - r2 * np.cos(theta2)
y4 = 4 + r2 * np.sin(theta2)
x5 = np.random.uniform(3, 4.5, 100)
y5 = 2.9 + 0.5 * (x5 - 3) + 0.1 * np.random.randn(100)
x = np.concatenate((x1, x2, x3, x4, x5))
y = np.concatenate((y1, y2, y3, y4, y5))
df7 = pd.DataFrame({'x': x, 'y': y})
df7['real labels'] = np.repeat([0, 0, 1, 2, 3], 100)


theta1 = np.random.uniform(0, 2*np.pi, 200)
r1 = 30 * np.sqrt(np.random.uniform(0, 1.2, 200))
x1 = 170 + r1 * np.cos(theta1)
y1 = 110 + r1 * np.sin(theta1)
theta2 = np.random.uniform(0, 2*np.pi, 200)
r2 = 30 * np.sqrt(np.random.uniform(0, 1.2, 200))
x2 = 320 + r2 * np.cos(theta2)
y2 = 110 + r2 * np.sin(theta2)
theta3 = np.random.uniform(0, 2*np.pi, 600)
r3 = 60 * np.sqrt(np.random.uniform(0, 2, 600))
x3 = 250 + r3 * np.cos(theta3)
y3 = 210 + r3 * np.sin(theta3)
x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))
df8 = pd.DataFrame({'x': x, 'y': y})
df8['real labels'] = np.repeat([0, 1, 2, 2, 2], 200)


x1 = np.random.uniform(150, 350, 200)
y1 = np.full(200, 110) + 2 * np.random.randn(200)
x2 = np.random.uniform(200, 285, 200)
y2 = np.full(200, 150) + 2 * np.random.randn(200)
x3 = np.random.uniform(200, 275, 200)
y3 = np.full(200, 190) + 2 * np.random.randn(200)
x4 = np.random.uniform(150, 350, 200)
y4 = np.full(200, 230) + 2 * np.random.randn(200)
x = np.concatenate((x1, x2, x3, x4))
y = np.concatenate((y1, y2, y3, y4))
df9 = pd.DataFrame({'x': x, 'y': y})
df9['real labels'] = np.repeat([0, 1, 2, 3], 200)


theta = np.linspace(0, 2*np.pi, 500)
r = theta
x_spiral1 = r * np.cos(theta) + np.random.normal(0, 0.05, len(theta)) + 1
y_spiral1 = r * np.sin(theta) + np.random.normal(0, 0.05, len(theta)) + 1 
x_spiral2 = r * np.cos(theta + 2*np.pi/3) + np.random.normal(0, 0.05, len(theta)) 
y_spiral2 = r * np.sin(theta + 2*np.pi/3) + np.random.normal(0, 0.05, len(theta)) + 0.5
x_spiral3 = r * np.cos(theta + 4*np.pi/3) + np.random.normal(0, 0.05, len(theta)) + 1 
y_spiral3 = r * np.sin(theta + 4*np.pi/3) + np.random.normal(0, 0.05, len(theta))
x = np.concatenate([x_spiral1, x_spiral2, x_spiral3])
y = np.concatenate([y_spiral1, y_spiral2, y_spiral3])
df10 = pd.DataFrame({'x': x, 'y': y})
df10['real labels'] = np.repeat([0, 1, 2], 500)


theta1 = np.random.uniform(0, 2*np.pi, 300)
r1 = 0.5 * np.sqrt(np.random.uniform(0, 1, 300))
x1 = -0.25 + r1 * np.cos(theta1)
y1 = -0.25 + r1 * np.sin(theta1)
theta2 = np.random.uniform(0, 2*np.pi, 300)
r2 = 0.05 * np.sqrt(np.random.uniform(0, 1, 300))
x2 = 0.9 + r2 * np.cos(theta2)
y2 = 0 + r2 * np.sin(theta2)
theta3 = np.random.uniform(0, 2*np.pi, 300)
r3 = 0.05 * np.sqrt(np.random.uniform(0, 1, 300))
x3 = 0.9 + r3 * np.cos(theta3)
y3 = -0.5 + r3 * np.sin(theta3)
x4 = np.random.uniform(-0.9, 0.9, 300)
y4 = np.full(300, 0.8) + 0.04 * np.random.randn(300)
x = np.concatenate((x1, x2, x3, x4))
y = np.concatenate((y1, y2, y3, y4))
df11 = pd.DataFrame({'x': x, 'y': y})
df11['real labels'] = np.repeat([0, 1, 2, 3], 300)


centers = [(0, 0), (20, 70), (45, 90), (75, 65), (90, 45), (92, 27)]
radii = [2.5, 3, 1, 0.5, 2.5, 1]
x = np.concatenate([np.full(50, cx) + radius * np.random.randn(50) for (cx, cy), radius in zip(centers, radii)])
y = np.concatenate([np.full(50, cy) + radius * np.random.randn(50) for (cx, cy), radius in zip(centers, radii)])
lines = [((-5, 40), (5, 40)), ((90, 70), (100, 70)), ((40, 5), (60, 5))]
for (x1, y1), (x2, y2) in lines:
    lx = np.linspace(x1, x2, 50)
    ly = np.linspace(y1, y2, 50) + 0.1 * np.random.randn(50) 
    x = np.concatenate((x, lx))
    y = np.concatenate((y, ly))
df12 = pd.DataFrame({'x': x, 'y': y})
df12['real labels'] = np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8], 50)


theta1 = np.random.uniform(-np.pi, 0, 333)
r1 = 0.2
x1 = 0.45 + r1 * np.cos(theta1)
y1 = 0.55 + r1 * np.sin(theta1)
theta2 = np.random.uniform(0, 2*np.pi, 333)
r2 = 0.05 * np.sqrt(np.random.uniform(0, 1, 333)) 
x2 = 0.35 + r2 * np.cos(theta2)
y2 = 0.6 + r2 * np.sin(theta2)
theta3 = np.random.uniform(0, 2*np.pi, 333)
r3 = 0.05 * np.sqrt(np.random.uniform(0, 1, 333)) 
x3 = 0.55 + r3 * np.cos(theta3)
y3 = 0.6 + r3 * np.sin(theta3)
x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))
df13 = pd.DataFrame({'x': x, 'y': y})
df13['real labels'] = np.repeat([0, 1, 2], 333)


theta = np.linspace(0, 2*np.pi, 100)
x_face = np.cos(theta) - 0.5
y_face = np.sin(theta) + 0.5
eye_radius = 0.05
eye_centers = [[-0.7, 0.75], [-0.3, 0.75]]
x_eyes = []
y_eyes = []
for center in eye_centers:
    theta = np.random.uniform(0, 2*np.pi, 100)
    x = center[0] + eye_radius * np.cos(theta)
    y = center[1] + eye_radius * np.sin(theta)
    x_eyes.extend(x)
    y_eyes.extend(y)
x_smile = np.linspace(-0.7, -0.3, 100)
y_smile = 0.2*np.cos(2*np.pi*(x_smile+0.5)) + 0.3
x = np.concatenate([x_face, x_eyes, x_smile])
y = np.concatenate([y_face, y_eyes, y_smile])
df14 = pd.DataFrame({'x': x, 'y': y})
df14['real labels'] = np.repeat([0, 1, 2, 3], 100)


x_line1 = np.linspace(-10, 10, 100)
y_line1 = np.zeros_like(x_line1) + np.random.uniform(0, 1, 100)
x_line2 = np.linspace(-10, 10, 100)
y_line2 = np.zeros_like(x_line2) - 5 + np.random.uniform(0, 1, 100)
circle_centers = [[10, 10], [15, 15], [10, 15], [15, 10]]
circle_radius = 1
points_per_circle = 100 
x_circles = []
y_circles = []
for center in circle_centers:
    r = circle_radius * np.sqrt(np.random.rand(points_per_circle))  
    theta = np.random.uniform(0, 2*np.pi, points_per_circle)
    x = center[0] + r * np.cos(theta) + np.random.normal(0, 0.5, points_per_circle) 
    y = center[1] + r * np.sin(theta) + np.random.normal(0, 0.5, points_per_circle)  
    x_circles.extend(x)
    y_circles.extend(y)
x = np.concatenate([x_line1, x_line2, x_circles])
y = np.concatenate([y_line1, y_line2, y_circles])
df15 = pd.DataFrame({'x': x, 'y': y})
df15['real labels'] = np.repeat([0, 1, 2, 3, 4, 5], 100)


points_main = 500
r_main = np.sqrt(np.random.rand(points_main))/2
theta_main = np.random.uniform(0, 2*np.pi, points_main)
x_main = r_main * np.cos(theta_main)
y_main = r_main * np.sin(theta_main)
points_small = 500
x_small1 = np.random.normal(0.6, 0.05, points_small)
y_small1 = np.random.normal(0.6, 0.05, points_small)
x_small2 = np.random.normal(0.6, 0.05, points_small)
y_small2 = np.random.normal(-0.6, 0.05, points_small)
x = np.concatenate([x_main, x_small1, x_small2])
y = np.concatenate([y_main, y_small1, y_small2])
df16 = pd.DataFrame({'x': x, 'y': y})
df16['real labels'] = np.repeat([0, 1, 2], 500)


theta = np.linspace(0, 2*np.pi, 1000)
r = 1 
x = r * np.cos(theta)
y = r * np.sin(theta)
mask1 = (theta >= 0) & (theta <= np.pi/2)
mask3 = (theta >= np.pi) & (theta <= 3*np.pi/2)
x = np.concatenate([x[mask1], x[mask3]])
y = np.concatenate([y[mask1], y[mask3]])
x += np.random.normal(0, 0.02, len(x))
y += np.random.normal(0, 0.02, len(y))
df17 = pd.DataFrame({'x': x, 'y': y})
df17['real labels'] = np.repeat([0, 1], 250)


x = np.linspace(-3, 3, 300)
y1 = np.zeros_like(x) + np.random.normal(0, 0.05, len(x))
y2 = np.ones_like(x) + np.random.normal(0, 0.05, len(x))
df18 = pd.DataFrame({
    'x': np.concatenate([x, x]),
    'y': np.concatenate([y1, y2])})
df18['real labels'] = np.repeat([0, 1], 300)


circle_centers = [[0, 0], [5, 6], [10, 10], [15, 15]]
circle_radius = 2
points_per_circle = 250
x_circles = []
y_circles = []
for center in circle_centers:
    r = circle_radius * np.sqrt(np.random.rand(points_per_circle)) 
    theta = np.random.uniform(0, 2*np.pi, points_per_circle)
    x = center[0] + r * np.cos(theta) + np.random.normal(0, 0.2, points_per_circle)  
    y = center[1] + r * np.sin(theta) + np.random.normal(0, 0.2, points_per_circle) 
    x_circles.append(x)
    y_circles.append(y)
x = np.concatenate(x_circles)
y = np.concatenate(y_circles)
df19 = pd.DataFrame({'x': x, 'y': y})
df19['real labels'] = np.repeat([0, 1, 2, 3], 250)


circle_centers = [[5, 5], [10, 10], [16, 16], [20, 20], [5, 20], [20, 5]]
circle_radius = 2
points_per_circle = 250
x_circles = []
y_circles = []
for center in circle_centers:
    r = circle_radius * np.sqrt(np.random.rand(points_per_circle)) 
    theta = np.random.uniform(0, 2*np.pi, points_per_circle)
    x = center[0] + r * np.cos(theta) + np.random.normal(0, 0.2, points_per_circle)  
    y = center[1] + r * np.sin(theta) + np.random.normal(0, 0.2, points_per_circle) 
    x_circles.append(x)
    y_circles.append(y)
x = np.concatenate(x_circles)
y = np.concatenate(y_circles)
df20 = pd.DataFrame({'x': x, 'y': y})
df20['real labels'] = np.repeat([0, 1, 2, 3, 4, 5], 250)


circle_centers = [[-10, 10], [-10, -10], [10, 10]]
circle_radius = 2
points_per_circle = 200 
x_circles = []
y_circles = []
for center in circle_centers:
    r = circle_radius * np.sqrt(np.random.rand(points_per_circle)) 
    theta = np.random.uniform(0, 2*np.pi, points_per_circle)
    x = center[0] + r * np.cos(theta) + np.random.normal(0, 0.2, points_per_circle) 
    y = center[1] + r * np.sin(theta) + np.random.normal(0, 0.2, points_per_circle) 
    x_circles.append(x)
    y_circles.append(y)
points_rectangle = 200
x_rectangle = np.random.uniform(0, 5, points_rectangle)
y_rectangle = np.random.uniform(-10, 10, points_rectangle)
x = np.concatenate([*x_circles, x_rectangle])
y = np.concatenate([*y_circles, y_rectangle])
df21 = pd.DataFrame({'x': x, 'y': y})
df21['real labels'] = np.repeat([0, 1, 2, 3], 200)


circle_centers = [[10, 10], [40, 60], [70, -15]]
circle_radius = 20
points_per_circle = 150
noise_std_dev = 5 
x_circles = []
y_circles = []
for center in circle_centers:
    r = circle_radius * np.sqrt(np.random.rand(points_per_circle)) 
    theta = np.random.uniform(0, 2*np.pi, points_per_circle)
    x = center[0] + r * np.cos(theta) + np.random.normal(0, noise_std_dev, points_per_circle)  
    y = center[1] + r * np.sin(theta) + np.random.normal(0, noise_std_dev, points_per_circle) 
    x_circles.append(x)
    y_circles.append(y)
x = np.concatenate(x_circles)
y = np.concatenate(y_circles)
df22 = pd.DataFrame({'x': x, 'y': y})
df22['real labels'] = np.repeat([0, 1, 2], 150)


cluster_centers = [[0, 5], [-5, 0], [0, -5]]
cluster_std_dev = 1.1
points_per_cluster = 200
x_clusters = []
y_clusters = []
for center in cluster_centers:
    x = np.random.normal(center[0], cluster_std_dev, points_per_cluster)
    y = np.random.normal(center[1], cluster_std_dev, points_per_cluster)
    x_clusters.append(x)
    y_clusters.append(y)
x = np.concatenate(x_clusters)
y = np.concatenate(y_clusters)
df23 = pd.DataFrame({'x': x, 'y': y})
df23['real labels'] = np.repeat([0, 1, 2], 200)


cluster_centers = [[3, 3.5]]
cluster_std_dev = 0.2
points_per_cluster = 100
x_clusters = []
y_clusters = []
for center in cluster_centers:
    x = np.random.normal(center[0], cluster_std_dev, points_per_cluster)
    y = np.random.normal(center[1], cluster_std_dev, points_per_cluster)
    x_clusters.append(x)
    y_clusters.append(y)
x3 = np.concatenate(x_clusters)
y3 = np.concatenate(y_clusters)
x1 = np.full(100, 2) + 0.1 * np.random.randn(100)
y1 = np.random.uniform(2, 4.5, 100)
x2 = np.random.uniform(2, 3.5, 100)
y2 = np.full(100, 2) + 0.1 * np.random.randn(100)
x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))
df24 = pd.DataFrame({'x': x, 'y': y})
df24['real labels'] = np.repeat([0, 1, 1], 100)


points_rectangle = 200
x_rectangle = np.random.uniform(-2, 2, points_rectangle)
y_rectangle = np.random.uniform(0, 3, points_rectangle)
df25 = pd.DataFrame({'x': x_rectangle, 'y': y_rectangle})
df25['real labels'] = np.repeat([0], 200)


dataframes = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14,
              df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25]
writer = pd.ExcelWriter(r'D:\Projects\In Progress Projects\Density-based CVI\Artificial Datasets.xlsx', engine='xlsxwriter')
for i, df in enumerate(dataframes, 1):
    df.to_excel(writer, sheet_name=f'df{i}')
writer.close()

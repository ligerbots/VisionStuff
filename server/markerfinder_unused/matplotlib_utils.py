import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches

field_width_in=360
field_height_in=180

marker_points_in={}
for x in range(11):
    for y in range(5):
        xx=(x+1)*30
        xn=str(x+1)

        yy=(y+1)*30
        yn=["E","D","C","B","A"][y]
        marker_points_in[yn+xn]=(xx,yy)

def init_field_plot():
    fig = plt.figure(figsize=(field_width_in/30, field_height_in/30)) # arbitrary scaling factor

    ax = fig.add_subplot()

    ax.spines['right'].set_position(('axes', 0))
    ax.spines['top'].set_position(('axes', 0))

    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('right')
    ax.grid(linestyle='-', linewidth='0.5')
    plt.axis([0, field_width_in, 0, field_height_in])

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(1, field_width_in/30+1, 1)*30)
    ax.set_yticks(np.arange(1, field_height_in/30+1, 1)*30)
    return(ax)

def addpt(ax,x,col):
    ax.plot(*x, "o"+col)

def addptname(ax,name,col):
    addpt(ax,marker_points_in[name],col)
    ax.annotate(name, marker_points_in[name])

def addrect(ax,a,b,color):
    rect = patches.Rectangle(a,b[0]-a[0],b[1]-a[1],linewidth=1,edgecolor=color,facecolor=color)
    ax.add_patch(rect)

def draw_gal_search_field(path):
    ax=init_field_plot()
    addrect(ax,(0,0),(30,180),"#bcd4bc")
    addrect(ax,(330,0),(360,180),"#d4c4c4")
    if(path=="A"):
        addptname(ax,"C3","r")
        addptname(ax,"D5","r")
        addptname(ax,"A6","r")
        addptname(ax,"E6","b")
        addptname(ax,"B7","b")
        addptname(ax,"C9","b")
    elif(path=="B"):
        addptname(ax,"B3","r")
        addptname(ax,"D5","r")
        addptname(ax,"B7","r")
        addptname(ax,"D6","b")
        addptname(ax,"B8","b")
        addptname(ax,"D10","b")
    return(ax)

def draw_autonav_field(path):
    ax=init_field_plot()

    if(path=="barrel"):
        addrect(ax,(0,60),(60,120),"#bcd4bc")
        addptname(ax,"D5","c")
        addptname(ax,"B8","c")
        addptname(ax,"D10","c")
    elif(path=="slalom"):
        addrect(ax,(0,0),(60,60),"#bcd4bc")
        addrect(ax,(0,60),(60,120),"#d4c4c4")
        addptname(ax,"D4","c")
        addptname(ax,"D5","c")
        addptname(ax,"D6","c")
        addptname(ax,"D7","c")
        addptname(ax,"D8","c")
        addptname(ax,"D10","c")
    elif(path=="bounce"):
        addrect(ax,(0,60),(60,120),"#bcd4bc")
        addrect(ax,(300,60),(360,120),"#d4c4c4")
        addptname(ax,"D3","c")
        addptname(ax,"E3","c")
        addptname(ax,"B4","c")
        addptname(ax,"B5","c")
        addptname(ax,"D5","c")
        addptname(ax,"B7","c")
        addptname(ax,"D7","c")
        addptname(ax,"B8","c")
        addptname(ax,"D8","c")
        addptname(ax,"A3","g")
        addptname(ax,"A6","g")
        addptname(ax,"A9","g")

    return(ax)

def drawline(a,b,style):
    plt.plot([a[0],b[0]],[a[1],b[1]],style)

def draw_robot(transform,fov_col="k"):
    robot_width_in=27
    robot_height_in=28

    pts=[
        [-robot_width_in/2,-robot_height_in/2],
        [robot_width_in/2,-robot_height_in/2],
        [robot_width_in/2,robot_height_in/2],
        [-robot_width_in/2,robot_height_in/2],
    ]
    fov_show_length=100
    half_fov=camera_matrix[0, 2]/camera_matrix[0, 0]
    pts+=[
        [0,0],
        [fov_show_length,math.tan(half_fov)*fov_show_length],
        [fov_show_length,-math.tan(half_fov)*fov_show_length],
    ]
    pts_trans=[
        np.array([*pt,1])@transform
        for pt in pts
    ]
    lines=[
        [pts_trans[0],pts_trans[1]],
        [pts_trans[1],pts_trans[2]],
        [pts_trans[2],pts_trans[3]],
        [pts_trans[3],pts_trans[0]],
    ]
    for line in lines:
        drawline(*np.array(line)[:,:2],"b-")
    lines_fov=[
        [pts_trans[4],pts_trans[5]],
        [pts_trans[4],pts_trans[6]],
    ]
    for line in lines_fov:
        drawline(*np.array(line)[:,:2],fov_col+"-")
        
def transform_mat(x,y,theta):
    return(np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta),  np.cos(theta), 0.],
        [            x,              y, 1.],
    ]))

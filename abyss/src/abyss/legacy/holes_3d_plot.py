import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
import abyss.dataparser as dp
from glob import glob
import os

def correctPerspective8B(path):
    if isinstance(path,str):
        img = cv2.imread(path)
    else:
        img = path
    height,width,_ = img.shape
    # manually selected points
    ins = np.float32([[36,8],
                    [6,339],
                    [484,10],
                    [507,345]])
    # corners of the image to correct to
    outs = np.float32([[0,0],
                      [0,height],
                      [width,0],
                      [width,height]])
    
    M = cv2.getPerspectiveTransform(ins,outs)
    warp = cv2.warpPerspective(img,M,(width,height))
    #cv2.imshow("orig",img)
    #cv2.imshow("warp",warp)

    return warp

def correctPerspective4B(path):
    if isinstance(path,str):
        img = cv2.imread(path)
    else:
        img = path
    height,width,_ = img.shape
    # manually selected points
    ins = np.float32([[15,9],
                    [1,538],
                    [743,17],
                    [756,536]])
    # corners of the image to correct to
    outs = np.float32([[0,0],
                      [0,height],
                      [width,0],
                      [width,height]])
    
    M = cv2.getPerspectiveTransform(ins,outs)
    warp = cv2.warpPerspective(img,M,(width,height))
    #cv2.imshow("orig",img)
    #cv2.imshow("warp",warp)

    return warp

def correctPerspectiveCorners(path):
    ''' Attempt to correct perspective by finding the corners of the block '''
    # read in source image
    if isinstance(path,str):
        img = cv2.imread(path)
    else:
        img = path
    # get size of the image
    height,width,_ = img.shape
    # perform canny edge to get the edges of the box
    edges = cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),180,230)
    # find the contours in the edges
    ct,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # get the 3 largest contours
    ct = sorted(ct,key=cv2.contourArea,reverse=True)[:3]
    # stack to form a single giant contour
    ct_arr = np.row_stack(ct)
    # fit convex hull around giant contour
    # hopefully this is the outer edge of the box
    hull = cv2.convexHull(ct_arr)
    # draw the convex hull on a blank image
    hull_mask = cv2.drawContours(np.zeros((height,width),np.uint8),[hull,],-1,255,1)
    # find 4 best features to track
    # hopefully corresponding to the edges of the box
    corners = cv2.goodFeaturesToTrack(hull_mask,4,0.01,10)
    corners = np.int0(corners)
    # convert results to a coordinate list
    ins = [i.ravel().tolist() for i in corners]
    # create list of corners corresponding to the corners of the image
    outs = np.float32([[0,0],
                      [0,height],
                      [width,0],
                      [width,height]])
    # sort corners based on which outer corner they're closest to
    ins.sort(key=lambda x : ((x[0]-width)**2 + (x[1]-height)**2)**0.5,reverse=True)
    ins = np.float32(ins)
    # find transform to correct perspective
    M = cv2.getPerspectiveTransform(ins,outs)
    # apply warp
    warp = cv2.warpPerspective(img,M,(width,height))
    # return resu;t
    return warp

def howCircular(ct):
    # find min enclosing circle around contour
    cent,rad = cv2.minEnclosingCircle(cv2.approxPolyDP(ct,3,True))
    # make dummy coordinates for a perfect circle
    theta = np.linspace(0,2*np.pi,ct.shape[0])
    xx = rad * np.cos(theta)
    yy = rad * np.sin(theta)
    # shift to centre
    xx += cent[0]
    yy += cent[1]
    # find percentage of values that are close
    close = np.isclose(ct.squeeze(),np.array([[x,y] for x,y in zip(xx,yy)]),rtol=0.01)
    return float(close[close==True].shape[0])/ct.shape[0]

def findHolesinImage(path='holes.png',correction='8b',show_image=True):
    # read in holes image
    if isinstance(path,str):
        holes = cv2.imread(path)
        if correction == '8b':
            holes = correctPerspective8B(holes)
        elif correction == '4b':
            holes = correctPerspectiveCorners(holes)
    else:
        holes = path
    if show_image:
        draw = holes.copy()
    # convert image to gray scale
    holes_gray = cv2.cvtColor(holes,cv2.COLOR_BGR2GRAY)
    # find the holes by searching for dark pixels
    if correction == '8b':
        mask = cv2.inRange(holes_gray,0,45)
        # clean up the mask by applying morph open
        mask_clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3)))
    elif correction == '4b':
        #mask = cv2.inRange(holes_gray,0,20)
        #mask = cv2.inRange(holes_gray,195,255)
        #mask = cv2.bitwise_not(mask)
        #holes_gray = cv2.bitwise_and(holes_gray,holes_gray,mask)
        mask = cv2.inRange(holes_gray,0,30)
        mask_clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5)))
        # manual additions to the mask for holes that aren't completely dark
        mask_clean = cv2.circle(mask_clean,(248,89),7,255,-1)
        mask_clean = cv2.circle(mask_clean,(452,219),7,255,-1)
        mask_clean = cv2.circle(mask_clean,(685,370),7,255,-1)
        mask_clean = cv2.circle(mask_clean,(145,89),7,255,-1)
        mask_clean = cv2.circle(mask_clean,(736,90),7,255,-1)
        ## mask to target area
        # target box
        mask_clean[200:,:] = 0
        # ignorre 'Life Test' text
        mask_clean[75:200,:50]=0
        mask_clean[:75,:]=0
    # show cleaned mask
    if show_image:
        cv2.imshow("mask",mask_clean)
    rows,cols = mask_clean.shape
    # force clean trailing pixels in the mask outside the box
    if correction == '8b':
        mask_clean[:,int(0.66*cols):] = 0
    # search for contours in the mask
    ct,_ = cv2.findContours(mask_clean,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #print(set([cv2.contourArea(cc) for cc in ct]))
    # remove small contours
    if correction == '8b':
        ct = list(filter(lambda x : cv2.contourArea(x)>140,ct))
    elif correction == '4b':
        #print(set([len(c) for c in ct]))
        #ct = list(filter(lambda x : cv2.contourArea(x)>10,ct))
        ct = list(filter(lambda x : cv2.minEnclosingCircle(cv2.approxPolyDP(x,3,True))[1]>=5,ct))
        #print(np.unique([howCircular(c) for c in ct],return_counts=True))
        #ct = list(filter(lambda x : howCircular(x)>=0.5,ct))
        #ct = list(filter(lambda x : len(x)>=10,ct))
    if show_image:
        # draw contours on image
        cv2.drawContours(draw,ct,-1,(255,0,0),2)

    centres = []
    width = []
    height = []
    # for each contour
    for c in ct:
        # extract the moments
        M = cv2.moments(c)
        # get coordinate for the centre
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # store coordinates
        centres.append((cX,cY))
        if show_image:
            cv2.circle(draw,(cX,cY),3,(255,255,255),-1)
            # draw circles on mask to try and clean up the contours
            #mask_clean = cv2.circle(mask_clean,(cX,cY),7,255,-1)
        # find bounding box
        x,y,w,h = cv2.boundingRect(c)
        width.append(w)
        height.append(h)

    # sort list by X (column) coordinate to get approximate order of holes
    if correction == '8b':
        centres.sort(key=lambda c : c[0])
        # manual adjustment
        # swap 67 and 68
        c67 = centres[66]
        c68 = centres[67]
        centres[66] = c68
        centres[67] = c67
    elif correction == '4b':
        sf = 35
        #centres.sort(key= lambda x : [(x[1]//sf)*sf,((cols-x[0])//10)*10],reverse=False)
        centres.sort(key=lambda x : [(x[1]//sf)*sf,((rows-x[0])//35)*35],reverse=False)
    #print("num centres: ",len(centres))
    if show_image:
        cv2.imshow("draw",draw)
    return centres,width,height,holes

def drawHoleOrder(path,centres,hide_text=False,hide_lines=False,use_arrows=False):
    ''' Draw lines connecting each fo the ordered centres and a text label for their centre ID '''
    if path is None:
        raise ValueError(f"Source image cannot be None! Received {path}!")
    if isinstance(path,str):
        draw = cv2.imread(path)
    else:
        draw = path
    # iterate over centres
    first = True
    for ci,(cA,cB) in enumerate(zip(centres,centres[1:]),start=1):
        # draw marker for first hole
        if first:
            draw = cv2.drawMarker(draw,cA,(0,255,0),cv2.MARKER_CROSS,5,3)
            first = False
        # draw text indicating hole index
        if not hide_text:
            draw = cv2.putText(draw,str(ci),(cA[0],cA[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1,cv2.LINE_AA)
        # draw line connecting neighbouring holes
        if not hide_lines:
            if use_arrows:
                cv2.arrowedLine(draw,cA,cB,(0,0,255),2,tipLength=0.02)
            else:
                draw = cv2.line(draw,cA,cB,(0,0,255),1)
        # draw a circle on the image showing the centre
        cv2.circle(draw,(cA[0],cA[1]), 2, (255,255,255),-1)
        cv2.circle(draw,(cB[0],cB[1]), 2, (255,255,255),-1)
        #draw = cv2.arrowedLine(draw,cA,cB,(0,0,255),1,tipLength=0.3)
    # draw blue marker for final centre
    draw = cv2.drawMarker(draw,centres[-1],(255,0,0),cv2.MARKER_CROSS,5,3)
    return draw

def plotTorqueBarsOnImage(holes=r"C:\Users\uos\Downloads\holes.png",setitec="8B life test\*.xls",plot_image=True):
    # load image
    holes = cv2.imread(holes)
    holes = correctPerspective8B(holes)
    # find holes in image
    centres,width,height = findHolesinImage(holes)
    #print(len(centres),len(width),len(height))
    # draw holes order
    draw = drawHoleOrder(holes,centres)
    # clip image to remove dead space
    holes = holes[:,:300]
    rows,cols,_ = holes.shape
    # create figure
    f = plt.figure(constrained_layout=True)
    # add 3d axes
    ax = f.add_subplot(projection='3d')
    # make x-y coordinates for pixels
    X1,Y1 = np.meshgrid(np.linspace(0,cols,cols),np.linspace(0,rows,rows))
    # plot the image as a surface
    # set face colors to those of the image
    stride=1
    if plot_image:
        ax.plot_surface(X1,Y1,np.zeros(X1.shape,X1.dtype)-0.1,rstride=stride,cstride=stride,linewidth=0,facecolors=img_as_float(holes[:,:,::-1]))
    # iterate over collection of centres, corresponding files sorted in order, widths and heights
    for cc,fn,w,h in zip(centres,sorted(glob(setitec),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])),width,height):
        # load file only extracting data
        data = dp.loadSetitecXls(fn,version="auto_data")
        # create a 3d bar where the height is the max torque height
        # coordinates are shifted so the bar is positioned in the centre of the circle
        # adjustment required as there is no align support
        ax.bar3d(float(cc[0])-w/2,float(cc[1])-h/2,0.0,w,h,data['I Torque (A)'].values.flatten().max(),shade=True,color='blue',alpha=0.8)
    # ensure that the z limit is set to 0
    ax.set_zlim(0,None)
    ax.set(xlabel="X",ylabel="Y",zlabel="Max Torque (A)",title="8B life test max torque")
    return f, draw

def get_energy_consumption_power(power, T):
    """
        Get energy consumption from a signal of power

        Inputs:
            power : Input signal of power
            T : period of the sampling

        Returns energy consumption
    """
    time = np.arange(0.0, len(power) * T, T, dtype='float16')
    time = np.around(time, 2)
    #power = power.fillna(0)
    energy = np.trapz(power, time[0:len(power)])

    return energy

def plotEnergyBarsOnImage(holes=r"C:\Users\uos\Downloads\holes.png",setitec="8B life test\*.xls",plot_image=True,T=1/100):
    # load image
    holes = cv2.imread(holes)
    holes = correctPerspective8B(holes)
    # find holes in image
    centres,width,height = findHolesinImage(holes)
    # draw holes order
    draw = drawHoleOrder(holes,centres)
    # clip image to remove dead space
    holes = holes[:,:300]
    rows,cols,_ = holes.shape
    # create figure
    f = plt.figure(constrained_layout=True)
    # add 3d axes
    ax = f.add_subplot(projection='3d')
    # make x-y coordinates for pixels
    X1,Y1 = np.meshgrid(np.linspace(0,cols,cols),np.linspace(0,rows,rows))
    # plot the image as a surface
    # set face colors to those of the image
    stride=1
    if plot_image:
        ax.plot_surface(X1,Y1,np.zeros(X1.shape,X1.dtype)-0.1,rstride=stride,cstride=stride,linewidth=0,facecolors=img_as_float(holes[:,:,::-1]))
    # iterate over collection of centres, corresponding files sorted in order, widths and heights
    for cc,fn,w,h in zip(centres,sorted(glob(setitec),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])),width,height):
        # load file only extracting data
        data = dp.loadSetitecXls(fn,version="auto_data")
        # create a 3d bar where the height is the energy
        # coordinates are shifted so the bar is positioned in the centre of the circle
        # adjustment required as there is no align support
        power = get_energy_consumption_power(data['Torque Power (W)'].values,T)
        ax.bar3d(float(cc[0])-w/2,float(cc[1])-h/2,0.0,w,h,power,shade=True,color='blue',alpha=0.8)
    # ensure that the z limit is set to 0
    ax.set_zlim(0,None)
    ax.set(xlabel="X",ylabel="Y",zlabel="Torque Energy (J)",title="8B life test Torque Energy")
    return f, draw

def plotContourfEnergyMap(holes="holes.png",setitec="8B life test\*.xls",plot_image=True,T=1/100,crop_deadspace=True):
    ''' Plot the energy of each hole as a colormap '''
    # load image
    holes = cv2.imread(holes)
    holes = correctPerspective8B(holes)
    # find holes in image
    centres,width,height = findHolesinImage(holes)
    # draw holes order
    draw = drawHoleOrder(holes,centres)
    cv2.imshow('draw',draw)
    # clip image to remove dead space
    if crop_deadspace:
        holes = holes[:,:300]
    rows,cols,_ = holes.shape
    # create figure
    f = plt.figure(constrained_layout=True)
    # add 3d axes
    ax = f.add_subplot()
    # make x-y coordinates for pixels
    X1,Y1 = np.meshgrid(np.linspace(0,cols,cols),np.linspace(0,rows,rows))
    Z = np.zeros(X1.shape)
    # plot the image as a surface
    # set face colors to those of the image
    stride=1
    # iterate over collection of centres, corresponding files sorted in order, widths and heights
    for cc,fn,w,h in zip(centres,sorted(glob(setitec),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])),width,height):
        # load file only extracting data
        data = dp.loadSetitecXls(fn,version="auto_data")
        # create a 3d bar where the height is the energy
        # coordinates are shifted so the bar is positioned in the centre of the circle
        # adjustment required as there is no align support
        power = get_energy_consumption_power(data['Torque Power (W)'].values,T)
        Z[int(cc[1]-h//2):int(cc[1]+h//2),int(cc[0]-w//2) : int(cc[0]+w//2)] = power
    # plot the energy as a contour
    cf = ax.contourf(X1,Y1,Z,cmap='hot')
    # invert y axis to put the origin in the top left corner as the coordinates are in image space
    ax.invert_yaxis()
    # add colorbar
    f.colorbar(cf)
    # add axis labels
    ax.set(xlabel="X-Axis",ylabel="Y-Axis",title="8B Life Test Energy")
    return f

def plotContourfTorqueMap(holes="holes.png",setitec="8B life test\*.xls",crop_deadspace=True):
    ''' Plot the max torque of each hole as a colormap '''
    # load image
    holes = cv2.imread(holes)
    holes = correctPerspective8B(holes)
    # find holes in image
    centres,width,height = findHolesinImage(holes)
    # draw holes order
    draw = drawHoleOrder(holes,centres)
    cv2.imshow('draw',draw)
    # clip image to remove dead space
    if crop_deadspace:
        holes = holes[:,:300]
    rows,cols,_ = holes.shape
    # create figure
    f = plt.figure(constrained_layout=True)
    # add 3d axes
    ax = f.add_subplot()
    # make x-y coordinates for pixels
    X1,Y1 = np.meshgrid(np.linspace(0,cols,cols),np.linspace(0,rows,rows))
    Z = np.zeros(X1.shape)
    # plot the image as a surface
    # set face colors to those of the image
    stride=1
    # iterate over collection of centres, corresponding files sorted in order, widths and heights
    for cc,fn,w,h in zip(centres,sorted(glob(setitec),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])),width,height):
        # load file only extracting data
        data = dp.loadSetitecXls(fn,version="auto_data")
        Z[int(cc[1]-h//2):int(cc[1]+h//2),int(cc[0]-w//2) : int(cc[0]+w//2)] = data['Torque Power (W)'].values.max()
    # plot max torque data as a contourf
    cf = ax.contourf(X1,Y1,Z,cmap='hot')
    # invert y axis to put origin in the top right hand corner as the coordinates are in image space
    ax.invert_yaxis()
    # add colormap
    f.colorbar(cf)
    # add labels
    ax.set(xlabel="X-Axis",ylabel="Y-Axis",title="8B Life Test Max Torque")
    return f

def dumpToCSV(holes="holes.png",setitec="8B life test\*.xls"):
    ''' Dump the hole data to a CSV file '''
    # load image
    holes = cv2.imread(holes)
    holes = correctPerspective8B(holes)
    # find holes in image
    centres,width,height = findHolesinImage(holes)
    ww,hh,_ = holes.shape
    ct = edgeContour(ww,hh)
    with open("8b-life-test-holes-data.csv",'w') as file:
        file.write("X,Y,Energy,Max Thrust,Distance\n")
        for cc,fn,w,h in zip(centres,sorted(glob(setitec),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])),width,height):
            # load file only extracting data
            data = dp.loadSetitecXls(fn,version="auto_data")
            # create a 3d bar where the height is the energy
            # coordinates are shifted so the bar is positioned in the centre of the circle
            # adjustment required as there is no align support
            power = get_energy_consumption_power(data['Torque Power (W)'].values,T)
            maxT = data['I Torque (A)'].values.flatten().max()
            file.write(','.join([str(x) for x in [cc[0],cc[1],power,maxT,cv2.pointPolygonTest(ct,(cc[0],cc[1]),measureDist=True)]]))
            file.write('\n')

def getJustHoleData(holes="holes.png",setitec="8B life test\*.xls"):
    '''
        Compile all data about holes into a single dictionary

        Stored data:
            Centres : Coordinte pair
            X : X-coordinte
            Y : Y-coordinate
            Energy : Energy expented to drill the hole
            MaxT : Max Torque when drilling hole
            Distances : Signed distance from hole centre until closest edge

        Inputs:
            holes : Input file path to image containing holes
            setitec : Path to Setitec XLS files

        Returns dictionary of data
    '''
    # load image
    holes = cv2.imread(holes)
    holes = correctPerspective8B(holes)
    # find holes in image
    centres,width,height = findHolesinImage(holes)
    # get shape of the image
    ww,hh,_ = holes.shape
    # create the edge contour data
    ct = edgeContour(ww,hh)
    holes_data = {'Centres': [],'X':[],'Y':[],'Energy':[],'MaxT':[],'Distances':[]}
    for cc,fn,w,h in zip(centres,sorted(glob(setitec),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])),width,height):
        # load file only extracting data
        data = dp.loadSetitecXls(fn,version="auto_data")
        # calculate energy
        power = get_energy_consumption_power(data['Torque Power (W)'].values,T)
        maxT = data['I Torque (A)'].values.flatten().max()
        holes_data['Centres'].append([cc[0],cc[1]])
        holes_data['X'].append(cc[0])
        holes_data['Y'].append(cc[1])
        holes_data['Energy'].append(power)
        holes_data['MaxT'].append(maxT)
        holes_dadta['Distances'].append(cv2.pointPolygonTest(ct,(cc[0],cc[1]),measureDist=True))
    return holes_data

def averageEnergyX():
    counts,bins = np.histogram(data['X'],10)
    avg =[ [] for _ in range(len(bins))]
    for xi,xx in enumerate(data['X']):
        ii = np.argmin(abs(bins-xx))
        avg[ii].append(data['Energy'][xi])
    avg = [np.mean(aa) for aa in avg]
    return avg

def model(x, a, b, c):
    return a * np.exp(b * x) + c

def fitExpToEnergy(y,x=None):
    if x is None:
        x = np.arange(0,len(y),1,dtype="int16")
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(lambda t,a,b: a+b*np.log(t),  x,  y,p0=[max(y)/10,min(y)],maxfev = 10000)
    A,B = popt
    f,ax = plt.subplots()
    ax.plot(x,y,'bx')
    ax.plot(x,B+A*np.log(x),'r')
    return f

def edgeContour(width,height):
    ''' Create contour for box around the edge of the image '''
    # top left to top right
    pts = [ [[x,0]] for x in range(width)]
    # top right to bottom right
    pts.extend([ [[width,y]] for y in range(height)])
    # bottom right to bottom left
    pts.extend([ [[width-x,height]] for x in range(width)])
    # bottom left to top left
    pts.extend([ [[0,height-y]] for y in range(height)])
    return np.array(pts,dtype="int32")

def findDistance(centres,ct):
    ''' Calculate signed distance from each centre to nearest point on contour '''
    return [cv2.pointPolygonTest(ct,tuple(cc),measureDist=True) for cc in centres]

def plotSignedDistanceEnergy(holes,setitec):
    ''' Plot the signed distance of each circle centre against the energy required to drill it '''
    data = getJustHoleData(holes,setitec)
    height,width,_ = cv2.imread(holes).shape
    ct = edgeContour(width,height)
    dist = findDistance(data['Centres'],ct)
    f,ax = plt.subplots()
    ax.plot(dist,data['Energy'],'bx')
    ax.set(xlabel="Signed Edge Distance (pixels)",ylabel="Max Energy",title="8B Life Test Signed Distance vs Energy")
    return f

def plotSignedDistanceHeatmap(holes="holes.png",centres=None,width=None,height=None,crop_deadspace=False,correction='8b',**kwargs):
    '''
        Plot a contourf of the distance of hole centres from the edge of the image

        Finds the holes in the image and find the signed distance to the edges of the image.
        Uses opencv pointPolygonTest to calculate the distance

        Inputs:
            holes : Input path to the image contianing the holes
            crop_deadspace : Flag to crop to column 300. Default False.

        Returns a figure of the signed distances as a colormap
    '''
    # load image
    if isinstance(holes,str):
        holes = cv2.imread(holes)
        # correct perspective
        if correction == '8b':
            holes = correctPerspective8B(holes)
        elif correction == '4b':
            holes = correctPerspectiveCorners(holes)

    # find holes in image if None were given
    if (centres is None) and (width is None) and (height is None):
        print("finding holes!")
        centres,width,height = findHolesinImage(holes)
        # clip image to remove dead space
        if crop_deadspace:
            holes = holes[:,:300]
    # get size of image
    rows,cols,_ = holes.shape
    # create contour point set around the edge of the image
    ct = edgeContour(cols,rows)
    # find distance between each circle centre of the edge of the image
    dist = findDistance(centres,ct)
    # create figure
    f = plt.figure(constrained_layout=True)
    # add 3d axes
    ax = f.add_subplot()
    # make x-y coordinates for pixels
    X1,Y1 = np.meshgrid(np.linspace(0,cols,cols),np.linspace(0,rows,rows))
    # create empty matrix
    Z = np.zeros(X1.shape)
    # iterate over collection of centres, corresponding files sorted in order, widths and heights
    # update the area of the circles with distance
    for cc,w,h,dd in zip(centres,width,height,dist):
        Z[int(cc[1]-h//2):int(cc[1]+h//2),int(cc[0]-w//2) : int(cc[0]+w//2)] = dd
    # plot the heatmap as a contour
    cf = ax.contourf(X1,Y1,Z,cmap='hot')
    # invert y axis to place origin top left as the source is image coordinates
    ax.invert_yaxis()
    # add colorbar
    f.colorbar(cf)
    # add axis labels
    ax.set(xlabel="X-Axis",ylabel="Y-Axis",title=kwargs.get('title',"Life Test Signed Distance"))
    # return figure
    return f

def groupCentresInStrips(centres,size,img,draw_on=True,mask_box=True,sort=True,show_box=True):
    '''
        Group centres based on their row location in blocks

        The area the holes are in is broken into section of height 'size'.
        The centres are placed into groups based on which strip they belong in
    '''
    # get shape of the image
    height,width,_ = img.shape
    if draw_on or show_box:
        # make a copy of the image to draw onb
        draw  = img.copy()
    # create list to hold groups
    groups = []
    # if the user is masking to a bounding box containing all centres
    if mask_box:
        # convert centres to a point list
        carr = np.expand_dims(np.array(centres),1)
        # find bounding box around poiints
        x,y,w,h = cv2.boundingRect(carr)
        # expand it a little to fully cover all boxes
        x -= 15
        y -= 15
        w += 30
        h += 29
        #print(w,h)
        if draw_on and show_box:
            cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)
        # generate the steps based on bounding box
        steps  = np.arange(y,y+h,size,dtype="uint16")
    # else generate the steps from the size of the image
    else:
        steps = np.arange(0,height,size,dtype="uint16")
    #print(steps)
    # iterate over the steps as pairs
    # check which centres fall into the target area
    for sa,sb in zip(steps,steps[1:]):
        # sort found centres from left to right
        # i.e. descending order of columns (Y)
        if sort:
            groups.append(sorted(list(filter(lambda x : (x[1]>=sa) & (x[1]<=sb),centres)),key=lambda x : x[0],reverse=True))
        # leave unsorted
        else:
            groups.append(list(filter(lambda x : (x[1]>=sa) & (x[1]<=sb),centres)))
    ll = groups[-1][-1]
    lb = groups[-1][-2]
    groups[-1][-1] = lb
    groups[-1][-2] = ll
    # if the user doesn't want the centres drawn
    if not draw_on:
        return groups,img
    # iterate over the group
    used_cols = set()
    for gi,gp in enumerate(groups,1):
        # generate a random color for the group
        col = tuple([int(c) for c in np.random.choice(range(256), size=3)])
        while col in used_cols:
            col = tuple([int(c) for c in np.random.choice(range(256), size=3)])
        # draw a circle for each centre using the target color
        for ca,cb in zip(gp,gp[1:]):
            cv2.circle(draw,ca,3,col,-1)
            cv2.circle(draw,cb,3,col,-1)
            cv2.arrowedLine(draw,ca,cb,(255,0,0),2,tipLength=0.3)
        used_cols.add(col)
    # return the groups and the drawing
    return groups,draw

def findExtraHolesLife4B(path,areaA=(455,490,703-455,517-490),areaB=(103,25,651-103,50-28),show_image=True):
    # read in holes image
    if isinstance(path,str):
        holes = cv2.imread(path)
        holes = correctPerspectiveCorners(holes)
    else:
        holes = path
    # convert image to gray scale
    holes_gray = cv2.cvtColor(holes,cv2.COLOR_BGR2GRAY)
    rows,cols = holes_gray.shape
    ### extra holes at the bottom of the plate
    if show_image:
        drawA = holes.copy()
    # make mask to target area at the bottom of the image
    x,y,w,h = areaA
    area_mask = cv2.rectangle(np.zeros(holes_gray.shape,dtype="uint8"),(x,y),(x+w,y+h),255,-1)
    holes_maskA = cv2.bitwise_and(holes_gray,holes_gray,mask=area_mask)
    # search for holes within the masked area
    mask = cv2.inRange(holes_maskA,0,30)
    mask_clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5)))
    # find contours
    ct,_ = cv2.findContours(mask_clean,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # remove largest contour as that's the outside of the masked area
    ct = sorted(ct,key=cv2.contourArea,reverse=True)[2:]
    if show_image:
        cv2.drawContours(drawA,ct,-1,(255,0,0),2)
    # create list of centres
    centresA = []
    widthA = []
    heightA = []
    # iterate over contours
    for c in ct:
        # extract the moments
        M = cv2.moments(c)
        # get coordinate for the centre
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # store coordinates
        centresA.append((cX,cY))
        # find bounding box around contour
        _,_,w,h = cv2.boundingRect(c)
        widthA.append(w)
        heightA.append(h)
        # draw circke on image marking centre
        if show_image:
            cv2.circle(drawA,(cX,cY),3,(255,255,255),-1)
    # order centres going from right to left
    ii = sorted(list(range(len(centresA))),key=lambda x : centresA[x][0],reverse=True)
    centresA = [centresA[x] for x in ii]
    widthA = [widthA[x] for x in ii]
    heightA = [heightA[x] for x in ii]
    #centresA = sorted(centresA,key=lambda x : x[0],reverse=True)

    #### extra holes at the top of the plate
    if show_image:
        drawB = holes.copy()
    x,y,w,h = areaB
    area_mask = cv2.rectangle(np.zeros(holes_gray.shape,dtype="uint8"),(x,y),(x+w,y+h),255,-1)
    holes_maskB = cv2.bitwise_and(holes_gray,holes_gray,mask=area_mask)
    # search for holes within the masked area
    mask = cv2.inRange(holes_maskB,0,30)
    mask_clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5)))
    # find contours
    ct,_ = cv2.findContours(mask_clean,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # remove largest contour as that's the outside of the masked area
    ct = sorted(ct,key=cv2.contourArea,reverse=True)[2:]
    if show_image:
        cv2.drawContours(drawB,ct,-1,(255,0,0),2)
    # create list of centres
    centresB = []
    widthB = []
    heightB = []
    # iterate over contours
    for c in ct:
        # extract the moments
        M = cv2.moments(c)
        # get coordinate for the centre
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # store coordinates
        centresB.append((cX,cY))
        _,_,w,h = cv2.boundingRect(c)
        widthB.append(w)
        heightB.append(h)
    # order centres going from right to left
    ii = sorted(range(len(centresB)),key=lambda x : centresB[x][0],reverse=True)
    #print(len(ii),len(centresB),len(widthB),len(heightB))
    centresB = [centresB[x] for x in ii]
    widthB = [widthB[x] for x in ii]
    heightB = [heightB[x] for x in ii]
    #centresB = sorted(centresB,key=lambda x : x[0],reverse=True)
    # remove 5th hole
    centresB.pop(5)
    widthB.pop(5)
    heightB.pop(5)
    # draw circke on image marking centre
    if show_image:
        for cX,cY in centresB:
            cv2.circle(drawB,(cX,cY),3,(255,255,255),-1)
    # if showing drawing
    if show_image:
        cv2.imshow("areaA",holes_maskA)
        cv2.imshow("drawA",drawA)
        cv2.imshow("areaB",holes_maskB)
        cv2.imshow("drawB",drawB)
    return centresA,widthA,heightA,centresB,widthB,heightB

def process4B():
    centres,width,height,holes=findHolesinImage('holes-4b.png',correction='4b',show_image=False)
    extraA,widthA,heightA,extraB,widthB,heightB = findExtraHolesLife4B(holes,show_image=False)
    print("extrasA ",extraA)
    print("extrasB ",extraB)
    groups,group_draw = groupCentresInStrips(centres,25,holes,False)
    cv2.imshow('group',group_draw)
    # flatten groups into a single list
    centres = [cc for gg in groups for cc in gg]
    # swap two specific ones in last group
    ll = centres[-2]
    centres[-2] = centres[-1]
    centres[-1] = ll
    # combine extra holes into a map of target locations
    # create blank array of known number of holes
    hashmap = 150*[None,]
    widthmap = 150*[None,]
    heightmap = 150*[None,]
    # start index at 0
    ci = 1
    # combine extras into single list
    extras= extraA[::-1] + extraB[::-1]
    extra_height = heightA[::-1] + heightB[::-1]
    extra_width = widthA[::-1] + widthB[::-1]
    # iterate over centres
    for cc,ww,hh in zip(centres,width,height):
        # every 10th hole
        # add one of the extras first
        if (ci!=0) and (ci%10)==0:
            hashmap[ci-1] = extras.pop()
            widthmap[ci-1] = extra_width.pop()
            heightmap[ci-1] = extra_height.pop()
            ci += 1
        # then add the centre in the next location
        hashmap[ci-1] = cc
        widthmap[ci-1] = ww
        heightmap[ci-1] = hh
        ci += 1
    hashmap[-1] = extras.pop()
    heightmap[-1] = extra_height.pop()
    widthmap[-1] = extra_width.pop()
    centres = hashmap
##    for c in centres:
##        draw_it = holes.copy()
##        cv2.circle(draw_it,c,3,(255,255,255),-1)
##        cv2.imshow("it",draw_it)
##        cv2.waitKey(0)
    # draw the order of the holes
    draw = drawHoleOrder(holes,hashmap,hide_text=True,use_arrows=True)
    cv2.imwrite('holes-4b-warp-draw.png',draw)
    cv2.imshow("order",draw)
    f = plotSignedDistanceHeatmap(holes,hashmap,widthmap,heightmap,title='4B Life Test Signed Distance')

def process8B():
    centres,widths,heights,holes=findHolesinImage('holes-8b.png',correction='8b',show_image=False)
    draw = drawHoleOrder(holes,centres,hide_text=True,use_arrows=True)
    print("num centres: ",len(centres))
    cv2.imshow("order",draw)
    f = plotSignedDistanceHeatmap(holes,centres,widths,height,title='8B Life Test Signed Distance')
    
    
if __name__ == "__main__":
    process4B()
   

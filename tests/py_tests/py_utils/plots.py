from math import floor
import plotly.graph_objects as go

# ================
# PLOT UTILS
# ================
def plot_img_segmentation_result(
    result , img_w=None , img_h=None ,
    plot_width=600 , plot_height=600 ,
    bidimensional_indices=False , int_rgb=False ):

    if "width" not in result and img_w is None:
        raise ValueError("The result does not contains 'width' and img_w is None." )
    elif "width" in result:
        width = result["width"]
    else:
        width = img_w

    if "height" not in result and img_h is None:
        raise ValueError("The result does not contains 'height' and img_h is None." )
    elif "height" in result:
        height = result["height"]
    else:
        height = img_h

    if not bidimensional_indices:
        indices_clusters = [
            [ (int(floor(l/width)) , int(l%width)) for l in cluster ]
            for cluster in result["clusters"] ]
    else:
        indices_clusters = result["clusters"]

    if not int_rgb:
        rgb_centroids = [
            [ int(floor(c[0]*255)) , int(floor(c[1]*255)) , int(floor(c[2]*255)) ]
            for c in result["centroids"]
        ]
    else:
        rgb_centroids = result["centroids"]

    traces = []
    for id,icluster,centroid in zip(range(len(indices_clusters)),indices_clusters,rgb_centroids):
        cluster_color = 'rgb(' + str(centroid[0]) + \
                           ',' + str(centroid[1]) + \
                           ',' + str(centroid[2]) + ')'
        traces.append( go.Scatter(
            x = [ i[1] for i in icluster ] ,
            y = [ height-i[0] for i in icluster ] ,
            mode = "markers" ,
            marker = dict(color=cluster_color) ,
            name = "cluster_" + str(id)
        ) )
    fig = go.Figure(data=traces)
    fig.update_layout( width=plot_width , height=plot_height )
    return fig

def plot_2D_clusters( data , res , title="" , width=800 , height=600 ):
    traces = [go.Scatter(
        x=[d[0] for d in data],
        y=[d[1] for d in data],
        mode="markers", name="data")]

    for i,cluster in zip(range(len(res['clusters'])) , res['clusters']):
        traces.append( go.Scatter(
            # x=[starts[ce][0] for ce in cluster ],
            # y=[starts[ce][1] for ce in cluster ],
            x=[data[ce][0] for ce in cluster ],
            y=[data[ce][1] for ce in cluster ],
            mode="markers" , name="cluster_"+str(i)
        ) )
        # traces.append( go.Scatter(
        #     x=[centroid[0]] , y=[centroid[1]] ,
        #     mode="markers" , marker=dict(color="red") ,
        #     name="centroid_"+str(i) ) )
    if 'centroids' in res:
        for i,centroid in zip(range(len(res['clusters'])) , res['centroids']):
            traces.append( go.Scatter(
                x=[centroid[0]] , y=[centroid[1]] ,
                mode="markers" , marker=dict(color="black") ,
                name="centroid_"+str(i) ) )

    fig = go.Figure( data=traces )
    fig.update_layout( width=width , height=height , title=title )
    fig.show()
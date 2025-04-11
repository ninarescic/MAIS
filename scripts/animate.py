import click    
import time
import pandas as pd
import graph_tool.all as gt

import cairo
from gi.repository import Gtk, GLib

from utils.config_utils import ConfigFile



@click.command()
@click.argument('filename')
@click.option('--nodes_file', default="../data/output/model/node_states.csv")
def main(filename, nodes_file):
    cf = ConfigFile()
    cf.load(filename)

    nodes_filename = cf.section_as_dict("GRAPH")["nodes"]
    edges_filename = cf.section_as_dict("GRAPH")["edges"]

    nodes = pd.read_csv(nodes_filename)[["id", "label", "sex"]]
    edges = pd.read_csv(edges_filename)[["vertex1", "vertex2", "probability", "intensity"]]

    g = gt.Graph(directed=False)
    node_label = g.new_vertex_property("string")
    node_sex = g.new_vertex_property("object")

    edge_proba = g.new_edge_property("float")
    edge_intensity = g.new_edge_property("float")

    lady = cairo.ImageSurface.create_from_png("lady.png")
    man = cairo.ImageSurface.create_from_png("man.png")
    dead = cairo.ImageSurface.create_from_png("zombie.png")

    node_state_I = g.new_vertex_property("bool")
    node_state_R = g.new_vertex_property("bool")

    df = pd.read_csv(nodes_file, index_col=0).transpose()

    def update_states(day):
        for i, row in nodes.iterrows():
            node_state_I[i] = df.loc[str(row["id"]), day] == 1
            if df.loc[str(row["id"]), day] == 2:
                node_sex[i] = dead



    for i, row in nodes.iterrows():
        v = g.add_vertex()
        node_label[v] = row["label"]
        node_sex[v] = lady if row["sex"] == "F" else man
        update_states(0)

    for i, row in edges.iterrows():
        v1, v2 = row["vertex1"], row["vertex2"]
        proba = row["probability"]
        intens = row["intensity"]

        e = g.add_edge(v1, v2, add_missing=True)
        edge_proba[e] = proba
        edge_intensity[e] = intens
        edge_width = proba*intens*2
        
        node_label[e.source()] = nodes.loc[nodes["id"] == v1, "label"].values[0]
        node_label[e.target()] = nodes.loc[nodes["id"] == v2, "label"].values[0]
        


    win = gt.GraphWindow(g, 
        pos = gt.sfdp_layout(g),
        geometry=(1200, 800),
        vertex_size=30,
        vertex_anchor=0,
        edge_color="gray",
        edge_pen_width =edge_width,
        edge_sloppy=True,
        vertex_surface=node_sex,
        vertex_color=[1.,1.,1.,0.],
        vertex_fill_color=[1.,1.,1.,0.],
        vertex_halo=node_state_I,
        vertex_halo_size=1.2,
        vertex_halo_color=[0.8, 0, 0, 0.6]
    )

    day = 0

    def update_state():
        global day
        update_states(day)

    
        win.graph.regenerate_surface()
        win.graph.queue_draw()
        day += 1
        time.sleep(2)
        if day == 10:
            return False
        return True


    cid = GLib.idle_add(update_state)

    # We will give the user the ability to stop the program by closing the window.
    win.connect("delete_event", Gtk.main_quit)

    # Actually show the window, and start the main loop.
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
""" This module implements some visualizations of CorEx representations.
"""

import os
from shutil import copyfile
from itertools import combinations
import numpy as np
import pylab
import networkx as nx
import matplotlib.pyplot as plt
import codecs
import seaborn as sns

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (255, 127, 14),
             (44, 160, 44), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


# Main visualization routines
def vis_rep(corex, data, row_label=None, column_label=None, prefix='corex_output', max_edges=200):
    """Various visualizations and summary statistics for a one layer representation"""
    if column_label is None:
        column_label = map(str, range(data.shape[1]))
    else:
        column_label = [extract_color(label)[0] for label in column_label]
    if row_label is None:
        row_label = map(str, range(len(data)))

    dual = (corex.moments['X_i Y_j'] * corex.moments['X_i Z_j']).T
    alpha = dual > 0.05

    print('Variable groups in summary/groups.txt')
    output_groups(corex.ws, corex.moments, alpha, dual, column_label, prefix=prefix)

    print("Latent factors for each sample in summary/labels.txt")
    labels = corex.transform(data)
    output_labels(labels, row_label, prefix=prefix)
    
    if hasattr(corex, "history"):
        print("Convergence of objective in summary/convergence.pdf")
        plot_convergence(corex.history, prefix=prefix)

    print 'Pairwise plots among high TC variables in "relationships"'
    plot_heatmaps(data, corex.mis, column_label, corex.transform(data), prefix=prefix)
    plot_top_relationships(data, corex, labels, column_label, prefix=prefix)


def output_groups(ws, moments, alpha, mis, column_label, thresh=0, prefix=''):
    tc = moments["TC"]
    tcs = moments["TCs"]
    add = moments["additivity"]
    dual = (moments['X_i Y_j'] * moments['X_i Z_j']).T
    f = safe_open(prefix + '/summary/groups.txt', 'w+')
    g = safe_open(prefix + '/summary/groups_no_overlaps.txt', 'w+')
    h = safe_open(prefix + '/summary/summary.txt', 'w+')
    h.write('Group, TC\n')
    m, nv = mis.shape
    f.write('variable, weight, MI\n')
    g.write('variable, weight, MI\n')
    for j in range(m):
        f.write('Group num: %d, TC(X;Y_j): %0.6f\n' % (j, tcs[j]))
        g.write('Group num: %d, TC(X;Y_j): %0.6f\n' % (j, tcs[j]))
        h.write('%d, %0.6f\n' % (j, tcs[j]))

        inds = np.where(alpha[j] > 0)[0]
        inds = inds[np.argsort(-np.abs(ws)[j][inds])]
        for ind in inds:
            f.write(column_label[ind] + ', {:.3f}, {:.3f}\n'.format(ws[j][ind], mis[j][ind]))
        inds = np.where(np.argmax(np.abs(ws), axis=0) == j)[0]
        inds = inds[np.argsort(-np.abs(ws)[j][inds])]
        for ind in inds:
            g.write(column_label[ind] + ', {:.3f}, {:.3f}\n'.format(ws[j][ind], mis[j][ind]))
    h.write('Total: {:f}\n'.format(np.sum(tcs)))
    h.write('The total of individual TCs should approximately equal the objective: {:f}\n'.format(tc))
    h.write('If not, this signals redundancy/synergy in the final solution (measured by additivity: {:f}'.format(add))
    f.close()
    g.close()
    h.close()


def output_labels(labels, row_label, prefix=''):
    f = safe_open(prefix + '/summary/labels.txt', 'w+')
    ns, m = labels.shape
    for l in range(ns):
        f.write(row_label[l] + ',' + ','.join(map(str, labels[l, :])) + '\n')
    f.close()


def plot_convergence(history, prefix='', prefix2=''):
    plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.plot(history["TC"], '-', lw=2.5, color=tableau20[0])
    x = len(history["TC"])
    y = np.max(history["TC"])
    plt.text(0.5 * x, 0.8 * y, "TC", fontsize=18, fontweight='bold', color=tableau20[0])

    if history.has_key("additivity"):
        plt.plot(history["additivity"], '-', lw=2.5, color=tableau20[1])
        plt.text(0.5 * x, 0.3 * y, "additivity", fontsize=18, fontweight='bold', color=tableau20[1])

    plt.ylabel('TC', fontsize=12, fontweight='bold')
    plt.xlabel('# Iterations', fontsize=12, fontweight='bold')
    plt.suptitle('Convergence', fontsize=12)
    filename = '{}/summary/convergence{}.pdf'.format(prefix, prefix2)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename, bbox_inches="tight")
    plt.close('all')
    return True



def plot_heatmaps(data, mis, column_label, cont, topk=10, prefix=''):
    cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
    m, nv = mis.shape
    for j in range(m):
        inds = np.argsort(- mis[j, :])[:topk]
        if len(inds) >= 2:
            plt.clf()
            order = np.argsort(cont[:,j])
            subdata = data[:, inds][order].T
            subdata -= np.nanmean(subdata, axis=1, keepdims=True)
            subdata /= np.nanstd(subdata, axis=1, keepdims=True)
            columns = [column_label[i] for i in inds]
            sns.heatmap(subdata, vmin=-3, vmax=3, cmap=cmap, yticklabels=columns, xticklabels=False, mask=np.isnan(subdata))
            filename = '{}/heatmaps/group_num={}.png'.format(prefix, j)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            plt.title("Latent factor {}".format(j))
            plt.yticks(rotation=0)
            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')
            #plot_rels(data[:, inds], map(lambda q: column_label[q], inds), colors=cont[:, j],
            #          outfile=prefix + '/relationships/group_num=' + str(j), latent=labels[:, j], alpha=0.1)


def plot_top_relationships(data, corex, labels, column_label, topk=5, prefix=''):
    dual = (corex.moments['X_i Y_j'] * corex.moments['X_i Z_j']).T
    alpha = dual > 0.04
    cy = corex.moments['ry']
    m, nv = alpha.shape
    for j in range(m):
        inds = np.where(alpha[j] > 0)[0]
        inds = inds[np.argsort(- dual[j][inds])][:topk]
        if len(inds) >= 2:
            if dual[j, inds[0]] > 0.1:
                factor = labels[:, j]
                title = '$Y_{%d}$' % j
            else:
                k = np.argmax(np.abs(cy[j]))
                if k == j:
                    k = np.argsort(-np.abs(cy[j]))[1]
                factor = corex.moments['X_i Z_j'][inds[0], j] * labels[:, j] + corex.moments['X_i Z_j'][inds[0], k] * labels[:, k]
                title = '$Y_{%d} + Y_{%d}$' % (j, k)
            plot_rels(data[:, inds], map(lambda q: column_label[q], inds), colors=factor,
                      outfile=prefix + '/relationships/group_num=' + str(j), title=title)


def plot_rels(data, labels=None, colors=None, outfile="rels", latent=None, alpha=0.8, title=''):
    ns, n = data.shape
    if labels is None:
        labels = map(str, range(n))
    ncol = 5
    nrow = int(np.ceil(float(n * (n - 1) / 2) / ncol))

    fig, axs = pylab.subplots(nrow, ncol)
    fig.set_size_inches(5 * ncol, 5 * nrow)
    pairs = list(combinations(range(n), 2))
    if colors is not None:
        colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

    for ax, pair in zip(axs.flat, pairs):
        diff_x = max(data[:, pair[0]]) - min(data[:, pair[0]])
        diff_y = max(data[:, pair[1]]) - min(data[:, pair[1]])
        ax.set_xlim([min(data[:, pair[0]]) - 0.05 * diff_x, max(data[:, pair[0]]) + 0.05 * diff_x])
        ax.set_ylim([min(data[:, pair[1]]) - 0.05 * diff_y, max(data[:, pair[1]]) + 0.05 * diff_y])
        ax.scatter(data[:, pair[0]], data[:, pair[1]], c=colors, cmap=pylab.get_cmap("jet"),
                       marker='.', alpha=alpha, edgecolors='none', vmin=0, vmax=1)

        ax.set_xlabel(shorten(labels[pair[0]]))
        ax.set_ylabel(shorten(labels[pair[1]]))

    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.scatter(data[:, 0], data[:, 1], marker='.')

    fig.suptitle(title, fontsize=16)
    pylab.rcParams['font.size'] = 12  #6
    # pylab.draw()
    # fig.set_tight_layout(True)
    pylab.tight_layout()
    pylab.subplots_adjust(top=0.95)
    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.set_visible(False)
    filename = outfile + '.png'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    fig.savefig(outfile + '.png')
    pylab.close('all')
    return True


# Hierarchical graph visualization utilities

def vis_hierarchy(corexes, column_label=None, max_edges=100, prefix=''):
    """Visualize a hierarchy of representations."""
    if column_label is None:
        column_label = map(str, range(corexes[0].mis.shape[1]))

    f = safe_open(prefix + '/summary/higher_layer_group_tcs.txt', 'w+')
    for j, corex in enumerate(corexes):
        f.write('At layer: %d, Total TC: %0.3f\n' % (j, corex.tc))
        f.write('Individual TCS:' + str(corex.tcs) + '\n')
        if hasattr(corex, "history"):
            plot_convergence(corex.history, prefix=prefix, prefix2=j)
    f.close()

    import textwrap
    column_label = map(lambda q: '\n'.join(textwrap.wrap(q, width=17, break_long_words=False)), column_label)

    #dual = (corex.moments['X_i Y_j'] * corex.moments['X_i Z_j']).T
    #alpha = dual > 0.04 # sieve.mis > (0.1 * np.max(sieve.mis, axis=1, keepdims=True)).clip(-np.log1p(-1. / sieve.n_samples) * 3)  # TODO: is that permanent?

    # Construct non-tree graph
    alphas = [(corex.moments['X_i Y_j'] * corex.moments['X_i Z_j']).T > 0.04 for corex in corexes]  # TODO: is that permanent?
    # weights = [alphas[k] * np.abs(corex.ws) / np.max(np.abs(corex.ws)) for k, corex in enumerate(corexes)]
    weights = [alphas[k] * np.abs(corex.ws) for k, corex in enumerate(corexes)]
    node_weights = [corex.tcs for corex in corexes]
    g = make_graph(weights, node_weights, column_label, max_edges=max_edges)

    # Display pruned version
    h = g.copy()  # trim(g.copy(), max_parents=max_parents, max_children=max_children)
    edge2pdf(h, prefix + '/graphs/graph_prune_' + str(max_edges), labels='label', directed=True, makepdf=True)

    # Display tree version
    tree = g.copy()
    tree = trim(tree, max_parents=1, max_children=False)
    edge2pdf(tree, prefix + '/graphs/tree', labels='label', directed=True, makepdf=True)

    return g


def neato(fname, position=None, directed=False):
    if directed:
        os.system(
            "sfdp " + fname + ".dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=False -o " + fname + "_sfdp.pdf")
        os.system(
            "sfdp " + fname + ".dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=True -o " + fname + "_sfdp_w_splines.pdf")
        return True
    if position is None:
        os.system("neato " + fname + ".dot -Tpdf -o " + fname + ".pdf")
        os.system("fdp " + fname + ".dot -Tpdf -o " + fname + "fdp.pdf")
    else:
        os.system("neato " + fname + ".dot -Tpdf -n -o " + fname + ".pdf")
    return True


def extract_color(label):
    import matplotlib

    colors = matplotlib.colors.cnames.keys()
    parts = label.split('_')
    for part in parts:
        if part in colors:
            parts.remove(part)
            return '_'.join(parts), part
    return label, 'black'


def edge2pdf(g, filename, threshold=0, position=None, labels=None, connected=True, directed=False, makepdf=True):
    #This function will takes list of edges and a filename
    #and write a file in .dot format. Readable, eg. by omnigraffle
    # OR use "neato file.dot -Tpng -n -o file.png"
    # The -n option says whether to use included node positions or to generate new ones
    # for a grid, positions = [(i%28,i/28) for i in range(784)]
    def cnn(node):
        #change node names for dot format
        if type(node) is tuple or type(node) is list:
            return u'n' + u'_'.join(map(unicode, node))
        else:
            return unicode(node)

    if connected:
        touching = list(set(sum([[a, b] for a, b in g.edges()], [])))
        g = nx.subgraph(g, touching)
        print 'non-isolated nodes,edges', len(list(g.nodes())), len(list(g.edges()))
    f = safe_open(filename + '.dot', 'w+')
    if directed:
        f.write("strict digraph {\n".encode('utf-8'))
    else:
        f.write("strict graph {\n".encode('utf-8'))
    #f.write("\tgraph [overlap=scale];\n".encode('utf-8'))
    f.write("\tnode [shape=point];\n".encode('utf-8'))
    for a, b, d in g.edges(data=True):
        if d.has_key('weight'):
            if directed:
                f.write(("\t" + cnn(a) + ' -> ' + cnn(b) + ' [penwidth=%.2f' % float(
                    np.clip(d['weight'], 0, 9)) + '];\n').encode('utf-8'))
            else:
                if d['weight'] > threshold:
                    f.write(("\t" + cnn(a) + ' -- ' + cnn(b) + ' [penwidth=' + str(3 * d['weight']) + '];\n').encode(
                        'utf-8'))
        else:
            if directed:
                f.write(("\t" + cnn(a) + ' -> ' + cnn(b) + ';\n').encode('utf-8'))
            else:
                f.write(("\t" + cnn(a) + ' -- ' + cnn(b) + ';\n').encode('utf-8'))
    for n in g.nodes():
        if labels is not None:
            if type(labels) == dict or type(labels) == list:
                thislabel = labels[n].replace(u'"', u'\\"')
                lstring = u'label="' + thislabel + u'",shape=none'
            elif type(labels) == str:
                if g.node[n].has_key('label'):
                    thislabel = g.node[n][labels].replace(u'"', u'\\"')
                    # combine dupes
                    #llist = thislabel.split(',')
                    #thislabel = ','.join([l for l in set(llist)])
                    thislabel, thiscolor = extract_color(thislabel)
                    lstring = u'label="%s",shape=none,fontcolor="%s"' % (thislabel, thiscolor)
                else:
                    weight = g.node[n].get('weight', 0.1)
                    if n[0] == 1:
                        lstring = u'shape=circle,margin="0,0",style=filled,fillcolor=black,fontcolor=white,height=%0.2f,label="Y%d"' % (
                            2 * weight, n[1])
                    else:
                        lstring = u'shape=point,height=%0.2f' % weight
            else:
                lstring = 'label="' + str(n) + '",shape=none'
            lstring = unicode(lstring)
        else:
            lstring = False
        if position is not None:
            if position == 'grid':
                position = [(i % 28, 28 - i / 28) for i in range(784)]
            posstring = unicode('pos="' + str(position[n][0]) + ',' + str(position[n][1]) + '"')
        else:
            posstring = False
        finalstring = u' [' + u','.join([ts for ts in [posstring, lstring] if ts]) + u']\n'
        #finalstring = u' ['+lstring+u']\n'
        f.write((u'\t' + cnn(n) + finalstring).encode('utf-8'))
    f.write("}".encode('utf-8'))
    f.close()
    if makepdf:
        neato(filename, position=position, directed=directed)
    return True


def shorten(s, n=12):
    if len(s) > 2 * n:
        return s[:n] + '..' + s[-n:]
    return s


def make_graph(weights, node_weights, column_label, max_edges=100):
    all_edges = np.hstack(map(np.ravel, weights))
    max_edges = min(max_edges, len(all_edges))
    w_thresh = np.sort(all_edges)[-max_edges]
    print 'weight threshold is %f for graph with max of %f edges ' % (w_thresh, max_edges)
    g = nx.DiGraph()
    max_node_weight = max([max(w) for w in node_weights])
    for layer, weight in enumerate(weights):
        m, n = weight.shape
        for j in range(m):
            g.add_node((layer + 1, j))
            g.node[(layer + 1, j)]['weight'] = 0.3 * node_weights[layer][j] / max_node_weight
            for i in range(n):
                if weight[j, i] > w_thresh:
                    if weight[j, i] > w_thresh / 2:
                        g.add_weighted_edges_from([( (layer, i), (layer + 1, j), 10 * weight[j, i])])
                    else:
                        g.add_weighted_edges_from([( (layer, i), (layer + 1, j), 0)])

    # Label layer 0
    for i, lab in enumerate(column_label):
        g.add_node((0, i))
        g.node[(0, i)]['label'] = lab
        g.node[(0, i)]['name'] = lab  # JSON uses this field
        g.node[(0, i)]['weight'] = 1
    return g


def trim(g, max_parents=False, max_children=False):
    for node in g:
        if max_parents:
            parents = list(g.successors(node))
            weights = [g.edge[node][parent]['weight'] for parent in parents]
            for weak_parent in np.argsort(weights)[:-max_parents]:
                g.remove_edge(node, parents[weak_parent])
        if max_children:
            children = g.predecessors(node)
            weights = [g.edge[child][node]['weight'] for child in children]
            for weak_child in np.argsort(weights)[:-max_children]:
                g.remove_edge(children[weak_child], node)
    return g


# Misc. utilities
def safe_open(filename, mode):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    return codecs.open(filename, mode, 'utf8')


if __name__ == '__main__':
    # Command line interface
    # Sample commands:
    # python vis_corex.py tests/test_data.csv
    import linearcorex as lc
    from time import time
    import csv
    import sys
    import traceback
    import cPickle
    import numpy.ma as ma
    from optparse import OptionParser, OptionGroup

    parser = OptionParser(usage="usage: %prog [options] data_file.csv \n"
                                "It is assumed that the first row and first column of the data CSV file are labels.\n"
                                "Use options to indicate otherwise.")
    group = OptionGroup(parser, "Input Data Format Options")
    group.add_option("-t", "--no_column_names",
                     action="store_true", dest="nc", default=False,
                     help="We assume the top row is variable names for each column. "
                          "This flag says that data starts on the first row and gives a "
                          "default numbering scheme to the variables (1,2,3...).")
    group.add_option("-f", "--no_row_names",
                     action="store_true", dest="nr", default=False,
                     help="We assume the first column is a label or index for each sample. "
                          "This flag says that data starts on the first column.")
    group.add_option("-m", "--missing",
                     action="store", dest="missing", type="float", default=-1e6,
                     help="Treat this value as missing data.")
    group.add_option("-d", "--delimiter",
                     action="store", dest="delimiter", type="string", default=",",
                     help="Separator between entries in the data, default is ','.")
    group.add_option("-g", "--gaussianize",
                     action="store", dest="gaussianize", type="string", default="standard",
                     help="Try gaussianize='outliers' if there are long tails.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "CorEx Options")
    group.add_option("-l", "--layers", dest="layers", type="string", default="2,1",
                     help="Specify number of units at each layer: 5,3,1 has "
                          "5 units at layer 1, 3 at layer 2, and 1 at layer 3")
    group.add_option("-w", "--max_iter",
                     action="store", dest="max_iter", type="int", default=10000,
                     help="Max number of iterations to use.")
    group.add_option("-a", "--additive",
                     action="store_false", dest="additive", default=True,
                     help="By default, we attempt to find non-synergistic solutions (better). -a will turn this off.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Computational Options")
    group.add_option("-n", "--gpu",
                     action="store_true", dest="gpu", default=False,
                     help="Try to use the gpu.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Output Options")
    group.add_option("-o", "--output",
                     action="store", dest="output", type="string", default="corex_output",
                     help="A directory to put all output files.")
    group.add_option("-v", "--verbose",
                     action="store", dest="verbose", type="int", default=0,
                     help="Print rich outputs while running (different levels of verbosity: 0,1,2).")
    group.add_option("-e", "--edges",
                     action="store", dest="max_edges", type="int", default=200,
                     help="Show at most this many edges in graphs.")
    group.add_option("-q", "--regraph",
                     action="store_true", dest="regraph", default=False,
                     help="Don't re-run corex, just re-generate outputs (perhaps with edges option changed).")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()
    if not len(args) == 1:
        print "Run with '-h' option for usage help."
        sys.exit()

    np.set_printoptions(precision=3, suppress=True)  # For legible output from numpy
    layers = map(int, options.layers.split(','))
    if layers[-1] != 1:
        layers.append(1)  # Last layer has one unit for convenience so that graph is fully connected.
    verbose = options.verbose

    #Load data from csv file
    filename = args[0]
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=options.delimiter)
        if options.nc:
            variable_names = None
        else:
            variable_names = reader.next()[(1 - options.nr):]
        sample_names = []
        data = []
        for row in reader:
            if options.nr:
                sample_names = None
            else:
                sample_names.append(row[0])
            data.append(row[(1 - options.nr):])

    try:
        X = np.array(data, dtype=float)  # Data matrix in numpy format
    except:
        print "Incorrect data format.\nCheck that you've correctly specified options " \
              "such as continuous or not, \nand if there is a header row or column.\n" \
              "Also, missing values should be specified with a numeric value (-1 by default).\n" \
              "Run 'python vis_corex.py -h' option for help with options."
        traceback.print_exc(file=sys.stdout)
        sys.exit()

    if verbose:
        print '\nData summary: X has %d rows and %d columns' % X.shape
        print 'Variable names are: ' + ','.join(map(str, list(enumerate(variable_names))))

    # Run CorEx on data
    if verbose:
        print 'Getting CorEx results'
    if not options.regraph:
        for l, layer in enumerate(layers):
            if verbose:
                print "Layer ", l
            if l == 0:
                t0 = time()
                corexes = [lc.Corex(n_hidden=layer, verbose=verbose, gaussianize=options.gaussianize,
                                    missing_values=options.missing, eliminate_synergy=options.additive,
                                    gpu=options.gpu,
                                    max_iter=options.max_iter).fit(X)]
                print 'Time for first layer: %0.2f' % (time() - t0)
                X_prev = X
            else:
                X_prev = corexes[-1].transform(X_prev)
                corexes.append(lc.Corex(n_hidden=layer, verbose=verbose, gaussianize=options.gaussianize,
                                        gpu=options.gpu,
                                        eliminate_synergy=options.additive, max_iter=options.max_iter).fit(X_prev))
        for l, corex in enumerate(corexes):
            # The learned model can be loaded again using ce.Corex().load(filename)
            print 'TC at layer %d is: %0.3f' % (l, corex.tc)
            cPickle.dump(corex, safe_open(options.output + '/layer_' + str(l) + '.dat', 'w'))
    else:
        corexes = [cPickle.load(open(options.output + '/layer_' + str(l) + '.dat')) for l in range(len(layers))]

    # This line outputs plots showing relationships at the first layer
    vis_rep(corexes[0], X, row_label=sample_names, column_label=variable_names, prefix=options.output)
    # This line outputs a hierarchical networks structure in a .dot file in the "graphs" folder
    # And it tries to compile the dot file into a pdf using the command line utility sfdp (part of graphviz)

    vis_hierarchy(corexes, column_label=variable_names, max_edges=options.max_edges, prefix=options.output)
""" This module implements some visualizations of CorEx representations.
"""

import os
from shutil import copyfile
from itertools import combinations
import numpy as np
import pylab
import networkx as nx


# Main visualization routines

def vis_rep(sieve, data, row_label=None, column_label=None, prefix='corex_output', max_edges=200):
    """Various visualizations and summary statistics for a one layer representation"""
    if column_label is None:
        column_label = map(str, range(data.shape[1]))
    if row_label is None:
        row_label = map(str, range(len(data)))
    column_label += ["Y%d" % j for j in range(sieve.m)]

    alpha = sieve.mis > 0  # TODO: is that permanent?
    print 'Groups in groups.txt'
    labels = sieve.transform(data)
    data = np.hstack([data, labels])
    output_groups(sieve.tcs, alpha, sieve.mis, column_label, prefix=prefix)
    output_labels(labels, row_label, prefix=prefix)
    if hasattr(sieve, "tc_history"):
        plot_convergence(sieve.tc_history, prefix=prefix)

    print 'Pairwise plots among high TC variables in "relationships"'
    plot_top_relationships(data, alpha, sieve.mis, column_label, labels, prefix=prefix)

    # vis_hierarchy(sieve, column_label, prefix=prefix, max_edges=max_edges)


def output_groups(tcs, alpha, mis, column_label, thresh=0, prefix=''):
    f = safe_open(prefix + '/text_files/groups.txt', 'w+')
    h = safe_open(prefix + '/text_files/summary.txt', 'w+')
    h.write('Group, TC\n')
    m, nv = mis.shape
    for j in range(m):
        f.write('Group num: %d, TC(X;Y_j): %0.6f\n' % (j, tcs[j]))
        h.write('%d, %0.6f\n' % (j, tcs[j]))

        inds = np.where(alpha[j] > 0)[0]
        inds = inds[np.argsort(-mis[j][inds])]
        for ind in inds:
            f.write(column_label[ind] + ', %0.6f\n' % mis[j][ind])
    h.write('Total: %0.6f' % np.sum(tcs))
    f.close()
    h.close()


def output_labels(labels, row_label, prefix=''):
    f = safe_open(prefix + '/text_files/cont_labels.txt', 'w+')
    ns, m = labels.shape
    for l in range(ns):
        f.write(row_label[l] + ',' + ','.join(map(str, labels[l, :])) + '\n')
    f.close()


def plot_convergence(tc_history, prefix=''):

    pylab.plot(tc_history)
    pylab.xlabel('# iterations')
    pylab.ylabel('$TC_L(X;Y)$')
    pylab.suptitle('Convergence of $TC_L(X;Y)$', fontsize=12)
    filename = prefix + '/text_files/convergence.pdf'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pylab.savefig(filename)
    pylab.close('all')
    return True


def plot_top_relationships(data, alpha, mis, column_label, cont, topk=5, prefix=''):
    m, nv = len(alpha), len(mis[0])
    for j in range(m):
        inds = np.where(alpha[j] > 0)[0]
        inds = inds[np.argsort(- mis[j][inds])][:topk]
        if len(inds) >= 2:
            plot_rels(data[:, inds], map(lambda q: column_label[q], inds), colors=cont[:, j],
                      outfile=prefix + '/relationships/group_num=' + str(j))


def plot_rels(data, labels=None, colors=None, outfile="rels", latent=None, alpha=0.5):
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
        ax.scatter(data[:, pair[0]], data[:, pair[1]], c=colors, cmap=pylab.get_cmap("jet"),
                       marker='.', alpha=alpha, edgecolors='none', vmin=0, vmax=1)

        ax.set_xlabel(shorten(labels[pair[0]]))
        ax.set_ylabel(shorten(labels[pair[1]]))

    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.scatter(data[:, 0], data[:, 1], marker='.')

    pylab.rcParams['font.size'] = 12  #6
    pylab.draw()
    #fig.set_tight_layout(True)
    fig.tight_layout()
    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.set_visible(False)
    filename = outfile + '.png'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    fig.savefig(outfile + '.png')
    pylab.close('all')
    return True


# Hierarchical graph visualization utilities

def vis_hierarchy(sieve, column_label, max_edges=200, prefix=''):
    """Visualize a hierarchy of representations."""
    import textwrap
    column_label = map(lambda q: '\n'.join(textwrap.wrap(q, width=20)), column_label)

    def f(j):
        if j < sieve.nv:
            return j
        else:
            return (1, j - sieve.nv)

    # Construct non-tree graph
    g = nx.DiGraph()
    max_node_weight = np.max(sieve.tcs)
    for i, c in enumerate(column_label):
        if i < sieve.nv:
            g.add_node(i)
            g.node[i]['weight'] = 1
            g.node[i]['label'] = c
            g.node[i]['name'] = c  # JSON uses this field
        else:
            g.add_node(f(i))
            g.node[f(i)]['weight'] = 0.33 * np.clip(sieve.tcs[i - sieve.nv] / max_node_weight, 0.33, 1)
        if i >= sieve.nv:
            g.add_weighted_edges_from([(f(j), (1, i - sieve.nv), sieve.mi_j(i - sieve.nv)[j]) for j in sieve.alpha[i - sieve.nv]])

    # Display pruned version
    h = g.copy()  # trim(g.copy(), max_parents=max_parents, max_children=max_children)
    h.remove_edges_from(sorted(h.edges(data=True), key=lambda q: q[2]['weight'])[:-max_edges])
    edge2pdf(h, prefix + '/graphs/graph_%d' % max_edges, labels='label', directed=True, makepdf=True)

    # Display tree version
    tree = g.copy()
    tree = trim(tree, max_parents=1, max_children=False)
    edge2pdf(tree, prefix + '/graphs/tree', labels='label', directed=True, makepdf=True)

    # Output JSON files
    try:
        import os
        print os.path.dirname(os.path.realpath(__file__))
        copyfile(os.path.dirname(os.path.realpath(__file__)) + '/tests/d3_files/force.html', prefix + '/graphs/force.html')
    except:
        print "Couldn't find 'force.html' file for visualizing d3 output"
    import json
    from networkx.readwrite import json_graph

    mapping = dict([(n, tree.node[n].get('label', str(n))) for n in tree.nodes()])
    tree = nx.relabel_nodes(tree, mapping)
    json.dump(json_graph.node_link_data(tree), safe_open(prefix + '/graphs/force.json', 'w+'))
    json.dump(json_graph.node_link_data(h), safe_open(prefix + '/graphs/force_nontree.json', 'w+'))

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
        print 'non-isolated nodes,edges', len(g.nodes()), len(g.edges())
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


def trim(g, max_parents=False, max_children=False):
    for node in g:
        if max_parents:
            parents = g.successors(node)
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
    return open(filename, mode)


if __name__ == '__main__':
    # Command line interface
    # Sample commands:
    # python vis_sieve.py tests/test_data.csv
    import sieve as sieve
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
    parser.add_option_group(group)

    group = OptionGroup(parser, "Sieve Options")
    group.add_option("-k", "--n_hidden", dest="n_hidden", type="int", default=2,
                     help="Latent factors take values 0, 1..k. Default k=2")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Output Options")
    group.add_option("-o", "--output",
                     action="store", dest="output", type="string", default="corex_output",
                     help="A directory to put all output files.")
    group.add_option("-v", "--verbose",
                     action="store_true", dest="verbose", default=False,
                     help="Print rich outputs while running.")
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

    verbose = options.verbose
    np.set_printoptions(precision=3, suppress=True)  # For legible output from numpy

    #Load data from csv file
    filename = args[0]
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ') #options.delimiter)
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
        Xm = ma.masked_equal(X, options.missing)
        mean_x = ma.mean(Xm, axis=0)[np.newaxis, :]
        X = np.where(X == options.missing, mean_x, X)
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

    # Run Sieve on data
    if verbose:
        print 'Getting Sieve results'
    if not options.regraph:
        s = sieve.Sieve(n_hidden=options.n_hidden, verbose=verbose, missing_values=options.missing).fit(X)
        cPickle.dump(s, open(options.output + '_sieve.dat', 'w'))
    else:
        s = cPickle.load(options.output + '_sieve.dat')

    # This line outputs plots showing relationships at the first layer
    vis_rep(s, X, row_label=sample_names, column_label=variable_names, prefix=options.output, max_edges=options.max_edges)
    # This line outputs a hierarchical networks structure in a .dot file in the "graphs" folder
    # And it tries to compile the dot file into a pdf using the command line utility sfdp (part of graphviz)

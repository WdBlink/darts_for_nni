'''
author: shell zhang @ Apr.25 2019
'''

import test_genotypes
from graphviz import Digraph
from PIL import Image
import os

height = 1000
width = 2000


class CustomVisualize:
    def __init__(self, output_path: str, project_name: str):
        self.output_path = output_path
        self.project_name = project_name
        self.normal_seq = []
        self.reduce_seq = []

    def plot(self, genotype, filename):
        g = Digraph(
            format='png',
            edge_attr=dict(fontsize='20', fontname="times"),
            node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                           penwidth='2', fontname="times"),
            engine='dot')
        g.body.extend(['rankdir=LR'])

        g.node("c_{k-2}", fillcolor='darkseagreen2')
        g.node("c_{k-1}", fillcolor='darkseagreen2')
        assert len(genotype) % 2 == 0
        steps = len(genotype) // 2

        for i in range(steps):
            g.node(str(i), fillcolor='lightblue')

        for i in range(steps):
            for k in [2 * i, 2 * i + 1]:
                op, j = genotype[k]
                if j == 0:
                    u = "c_{k-2}"
                elif j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j - 2)
                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")

        g.node("c_{k}", fillcolor='palegoldenrod')
        for i in range(steps):
            g.edge(str(i), "c_{k}", fillcolor="gray")

        g.render(filename, view=False)

        if filename == 'normal':
            self.normal_seq.append(Image.open(filename + '.png').resize((width, height), Image.ANTIALIAS))
        else:
            self.reduce_seq.append(Image.open(filename + '.png').resize((width, height), Image.ANTIALIAS))

        os.remove(filename)

    def output(self):
        img_normal = self.normal_seq[0]
        img_normal.save(os.path.join(self.output_path, '{0}_normal.gif'.format(self.project_name)), save_all=True,
                        append_images=self.normal_seq, loop=1000, duration=1000)

        img_reduce = self.reduce_seq[0]
        img_reduce.save(os.path.join(self.output_path, '{0}_reduce.gif'.format(self.project_name)), save_all=True,
                        append_images=self.reduce_seq, loop=1000, duration=1000)


if __name__ == '__main__':
    vis = CustomVisualize('C:\\Users\\Tabulator\\Desktop\\custom_visualize', 'test')
    todo = [test_genotypes.AmoebaNet, test_genotypes.DARTS_V1, test_genotypes.DARTS_V2, test_genotypes.NASNet]

    for item in todo:
        vis.plot(genotype=item.normal, filename='normal')
        vis.plot(genotype=item.reduce, filename='reduce')

    vis.output()

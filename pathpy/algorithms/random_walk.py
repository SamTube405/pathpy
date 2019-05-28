"""
Algorithms to calculate shortest paths and distances in higher-order networks and paths.
"""
# -*- coding: utf-8 -*-

#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2018 Ingo Scholtes, ETH Zürich/Universität Zürich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:
#
#    E-mail: scholtes@ifi.uzh.ch
#    Web:    http://www.ingoscholtes.net
from functools import singledispatch
from collections import defaultdict

from pathpy.utils import Log, Severity
from pathpy.utils import PathpyNotImplemented
from pathpy.classes import TemporalNetwork
from pathpy.classes import Network
from pathpy.classes import HigherOrderNetwork
import numpy as _np

__all__ = ['generate_walk']

@singledispatch
def generate_walk(network, l=100, start_node=None):
    """
    Generate a random walk trajectory of a given length, based on
    a weighted/directed/undirected network, temporal network, or
    higher-order network.

    Parameters:
    -----------
    network: Network, TemporalNetwork, HigherOrderNetwork
        The temporal, first-order, or higher-order network, which
        will be used to randomly generate a walk through a network.
    l: int
        The (maximum) length of the walk to be generated. If a node
        with out-degree zero is encountered, the walk is terminated
        even if l has not been reached.
    start_node: str
        The (higher-order) node in which the random walk will be started.
        Default is None, in which case a random start node will be chosen.
    """
    T = network.transition_matrix().todense().transpose()
    idx_map = network.node_to_name_map()
    nodes = _np.array([v for v in network.nodes])

    itinerary = []

    if start_node is None:
        start_node = _np.random.choice(nodes)

    # choose random start node
    itinerary.append(start_node)
    for j in range(l):
        # get transition probability vector T[idx ->  . ]
        prob = _np.array(T[idx_map[itinerary[-1]], :])[0, :]
        nz = prob.nonzero()[0]
        # make one random transition
        if nz.shape[0] > 0:
            next_node = _np.random.choice(a=nodes[nz], p=prob[nz])
            # add node to path
            itinerary.append(next_node)
        else: # no neighbor
            break
    return itinerary


@generate_walk.register(HigherOrderNetwork)
def _temporal_walk(higher_order_net, l=100, start_node=None):

    T = higher_order_net.transition_matrix().todense().transpose()
    idx_map = higher_order_net.node_to_name_map()
    nodes = _np.array([v for v in higher_order_net.nodes])

    itinerary = []

    if start_node is None:
        start_node = _np.random.choice(nodes)
    last = start_node

    # choose random start node
    for x in higher_order_net.higher_order_node_to_path(start_node):
        itinerary.append(x)
    for j in range(l):
        # get transition probability vector T[idx ->  . ]
        prob = _np.array(T[idx_map[last], :])[0, :]
        nz = prob.nonzero()[0]
        # make one random transition
        if nz.shape[0] > 0:
            next_node = _np.random.choice(a=nodes[nz], p=prob[nz])
            # add node to path
            itinerary.append(higher_order_net.higher_order_node_to_path(next_node)[-1])
            last = next_node
        else: # no neighbor
            break
    return itinerary


@generate_walk.register(TemporalNetwork)
def _temporal_walk(tempnet, l=100, start_node=None):

    itinerary = []
    if start_node is None:
        current_node = _np.random.choice(tempnet.nodes)
    else:
        current_node = start_node
    itinerary.append(current_node)
    steps = 0
    for t in tempnet.ordered_times:
        prev_node = current_node
        # find possible targets in time t
        targets = set()
        for (v, w, time) in tempnet.time[t]:
            if v == current_node:
               targets.add(w)
        # move to random target
        if targets:
            current_node = _np.random.choice(list(targets))
            steps += 1
        # add currently visited node
        if current_node != prev_node:
            itinerary.append(current_node)
        if steps == l:
            break
    return itinerary


def _temporal_walk_layer(tempnet1, tempnet2, layernet, start_node, start_layer):
    def temporal_edge_filter(v, landmark, tempnet):
        new_targets = {}

        # get the activities given the node
        v_acts_timeline = tempnet.activities[v]
        act_x_mark = None
        # filter the activity timeline by the landmark
        for act_mark in v_acts_timeline:
            if act_mark >= landmark:
                act_x_mark = act_mark
                break;

        # find possible targets in tempnet
        if act_x_mark is not None:
            delay = act_x_mark - landmark
            v_targets = tempnet.sources[act_x_mark][v]
            for v_target_tuple in v_targets:
                new_targets[v_target_tuple[1]] = delay

        return new_targets

    ## pick the start layer
    if start_layer == 1:
        tempnet=tempnet1
    elif start_layer == 2:
        tempnet=tempnet2

    itinerary = []
    if start_node is None:
        current_node = _np.random.choice(tempnet.nodes)
    else:
        current_node = start_node

    itinerary.append(current_node)

    current_acts_timeline = tempnet.activities[current_node]

    ## check whether the given start node has associated activities
    if len(current_acts_timeline) > 0:

        ## pick a random activity
        ## this could be improved, where we can favor the recent edges
        ## e.g., pick edges based on a power law distribution
        current_start_t = _np.random.choice(current_acts_timeline)

        ## initial cross layer targets
        start_targets = list(layernet.successors[current_node])

        if len(start_targets) > 0:
            ## pick up a random cross target
            start_target = _np.random.choice(start_targets)

            steps = 0

            ## get all time-steps in the multiplex network
            time_steps = sorted(list(set(tempnet1.ordered_times + tempnet2.ordered_times)))

            for t in time_steps:
                ## initial jump
                if t >= current_start_t:
                    prev_node = current_node
                    # keep the parent-child relationships
                    # important when the walker cross the layer
                    # target is the key, and the value is the parent
                    targets = {}

                    # find possible targets in tempnet1
                    for (v, w, time) in tempnet1.time[t]:
                        if v == current_node:
                            targets[w]=w

                    # find possible targets in tempnet2
                    for (v, w, time) in tempnet2.time[t]:
                        if v == current_node:
                            targets[w]=w

                    ## adding cross-layer targets
                    layer_targets = list(layernet.successors[current_node])

                    ## you can control navigability here
                    if len(layer_targets) > 0:
                        for layer_target in layer_targets:

                            # find possible edges originated from cross-layer targets
                            # only forward temporal edges are selected
                            layer=layernet.nodes[layer_target]['layer']
                            if layer == 1:
                                _new_targets = temporal_edge_filter(layer_target, t, tempnet1)
                            elif layer == 2:
                                _new_targets = temporal_edge_filter(layer_target, t, tempnet2)

                            # update the target list
                            for _new_target, _new_target_delay in _new_targets.items():
                                targets[_new_target] = layer_target

                    # move to random target
                    if targets:
                        current_node = _np.random.choice(list(targets.keys()))
                        current_node_parent = targets[current_node]

                        # add currently visited node
                        if current_node != prev_node:
                            # if the target is a cross-layer target, add the parent
                            if current_node != current_node_parent:
                                itinerary.append(current_node_parent)

                            # always add the child
                            itinerary.append(current_node)

                            steps += 1

    return itinerary

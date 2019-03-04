#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 08:52:32 2018

@author: Moritz Heusinger <moritz.heusinger@fhws.de>
"""

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from rslvq import RSLVQ

"""1. Create stream"""
stream = ConceptDriftStream(stream=MIXEDGenerator(random_state=112, classification_function=0), 
                            drift_stream=MIXEDGenerator(random_state=112, 
                                                          classification_function=1),
                            random_state=None,
                            position=25000,
                            width=25000)

stream.prepare_for_use()

"""2. Create classifier"""
clf = [
       RSLVQ(prototypes_per_class=1, sigma=1.0, decay_rate=0.999),
       RSLVQ(prototypes_per_class=1, sigma=1.0, gradient_ascent='SGA')
       ]

"""3. Setup evaluator"""
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=1,
                                max_samples=100000,
                                metrics=['accuracy', 'kappa'],
                                output_file=None)

"""4. Run evaluator"""
evaluator.evaluate(stream=stream, model=clf, model_names=['ADADELTA', 'SGD'])
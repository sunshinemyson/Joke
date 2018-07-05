# //=============================================================================
# //  @@-COPYRIGHT-START-@@
# //
# //  Copyright 2015-2018 Qualcomm Technologies, Inc. All rights reserved.
# //  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
# //
# //  The party receiving this software directly from QTI (the "Recipient")
# //  may use this software as reasonably necessary solely for the purposes
# //  set forth in the agreement between the Recipient and QTI (the
# //  "Agreement"). The software may be used in source code form solely by
# //  the Recipient's employees (if any) authorized by the Agreement. Unless
# //  expressly authorized in the Agreement, the Recipient may not sublicense,
# //  assign, transfer or otherwise provide the source code to any third
# //  party. Qualcomm Technologies, Inc. retains all ownership rights in and
# //  to the software
# //
# //  This notice supersedes any other QTI notices contained within the software
# //  except copyright notices indicating different years of publication for
# //  different portions of the software. This notice does not supersede the
# //  application of any third party copyright notice to that third party's
# //  code.
# //
# //  @@-COPYRIGHT-END-@@
# //=============================================================================
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


real_div_sequence = GraphSequence([
    ConverterSequenceNode('root', ['RealDiv']),
    NonConsumableConverterSequenceNode('a', ['?']),
    NonConsumableConverterSequenceNode('b', ['?'])
])
real_div_sequence.set_inputs('root', ['a', 'b'])
real_div_sequence.set_outputs(['root'])

identity_sequence = GraphSequence([
    ConverterSequenceNode('root', ['Identity']),
    NonConsumableConverterSequenceNode('any', ['?']),
])
identity_sequence.set_inputs('root', ['any'])
identity_sequence.set_outputs(['root'])

placeholder_with_default_sequence = GraphSequence([
    ConverterSequenceNode('root', ['PlaceholderWithDefault']),
    NonConsumableConverterSequenceNode('any', ['?']),
])
placeholder_with_default_sequence.set_inputs('root', ['any'])
placeholder_with_default_sequence.set_outputs(['root'])

ignored_sequence_1 = GraphSequence([
    ConverterSequenceNode('root', ['Pack']),
    ConverterSequenceNode('a', ['Add']),
    ConverterSequenceNode('b', ['Add']),
    ConverterSequenceNode('c', ['Mul']),
    ConverterSequenceNode('d', ['Mul']),
    ConverterSequenceNode('e', ['?']),
    ConverterSequenceNode('f', ['?']),
    ConverterSequenceNode('g', ['?']),
    ConverterSequenceNode('h', ['?']),
    ConverterSequenceNode('i', ['?']),
    ConverterSequenceNode('j', ['?']),
    ConverterSequenceNode('k', ['?']),
    ConverterSequenceNode('l', ['?'])
])
ignored_sequence_1.set_inputs('root', ['a', 'b', 'e', 'f'])
ignored_sequence_1.set_inputs('a', ['c', 'g'])
ignored_sequence_1.set_inputs('b', ['d', 'h'])
ignored_sequence_1.set_inputs('c', ['i', 'j'])
ignored_sequence_1.set_inputs('d', ['k', 'l'])
ignored_sequence_1.set_outputs(['root'])

ignored_sequence_2 = GraphSequence([
    ConverterSequenceNode('root', ['Pack']),
    ConverterSequenceNode('a', ['Mul']),
    ConverterSequenceNode('b', ['Mul']),
    ConverterSequenceNode('e', ['?']),
    ConverterSequenceNode('f', ['?'])
])
ignored_sequence_2.set_inputs('root', ['a', 'b', 'e', 'f'])
ignored_sequence_2.set_outputs(['root'])

dropout_cell_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('is_training/read', ['?']),
    ConverterSequenceNode('Dropout/cond/Switch', ['Switch']),
    NonConsumableConverterSequenceNode('Dropout/cond/switch_t', ['Identity']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/random_uniform/min', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/random_uniform/max', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/Shape', ['Const']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/sub', ['Sub']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/RandomUniform', ['RandomUniform']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/mul', ['Mul']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform', ['Add']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/keep_prob', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/pred_id', ['Identity']),
    ConverterSequenceNode('Dropout/cond/dropout/add', ['Add']),
    ConverterSequenceNode('Dropout/cond/dropout/div/Switch', ['Switch']),
    ConverterSequenceNode('Dropout/cond/dropout/Floor', ['Floor']),
    ConverterSequenceNode('Dropout/cond/dropout/div', ['RealDiv']),
    ConverterSequenceNode('Dropout/cond/dropout/mul', ['Mul']),
    ConverterSequenceNode('Dropout/cond/Switch_1', ['Switch']),
    ConverterSequenceNode('Dropout/cond/Merge', ['Merge']),
    NonConsumableConverterSequenceNode('stub_20', ['?']),
    NonConsumableConverterSequenceNode('stub_25', ['?']),
])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/add',
                              ['Dropout/cond/dropout/keep_prob', 'Dropout/cond/dropout/random_uniform'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/Floor', ['Dropout/cond/dropout/add'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/mul',
                              ['Dropout/cond/dropout/random_uniform/RandomUniform',
                               'Dropout/cond/dropout/random_uniform/sub'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/div',
                              ['Dropout/cond/dropout/div/Switch', 'Dropout/cond/dropout/keep_prob'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform', ['Dropout/cond/dropout/random_uniform/mul',
                                                                      'Dropout/cond/dropout/random_uniform/min'])
dropout_cell_sequence.set_inputs('Dropout/cond/Switch', ['stub_20', 'is_training/read'])
dropout_cell_sequence.set_inputs('Dropout/cond/pred_id', ['is_training/read'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/RandomUniform',
                              ['Dropout/cond/dropout/Shape'])
dropout_cell_sequence.set_inputs('Dropout/cond/Merge', ['Dropout/cond/Switch_1', 'Dropout/cond/dropout/mul'])
dropout_cell_sequence.set_inputs('Dropout/cond/switch_t', ['Dropout/cond/Switch'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/mul',
                              ['Dropout/cond/dropout/div', 'Dropout/cond/dropout/Floor'])
dropout_cell_sequence.set_inputs('Dropout/cond/Switch_1', ['stub_25', 'Dropout/cond/pred_id'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/sub',
                              ['Dropout/cond/dropout/random_uniform/max',
                               'Dropout/cond/dropout/random_uniform/min'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/div/Switch', ['stub_25', 'Dropout/cond/pred_id'])
dropout_cell_sequence.set_outputs(['Dropout/cond/Merge'])

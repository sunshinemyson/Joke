#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2018 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
#==============================================================================

class Translation(object):
    def __init__(self):
        self.indexed_methods = {}

    def apply_method(self, method_name, *args):
        return self.indexed_methods[method_name](*args)

    def index_method(self, method_name, method):
        self.indexed_methods[method_name] = method

    def has_indexed_method(self, method_name):
        return method_name in self.indexed_methods

class TranslationBank(object):
    def __init__(self):
        # string type name -> translation
        # the same value may exist for multiple keys.
        self.translations = {}

    def __get_translation(self, op_type):
        if not op_type in self.translations:
            raise KeyError("No translation registered for op type %s" % op_type)
        return self.translations[op_type]

    def apply_specific(self, op_type, method_name, *args):
        translation = self.__get_translation(op_type)
        if not translation.has_indexed_method(method_name):
            raise KeyError("Translation for '%s' does not define an indexed method '%s'" % (op_type, method_name))
        return translation.apply_method(method_name, *args)

    def apply_partial(self, method_name, graph, *args):
        for node in graph.list_nodes():
            translation = self.__get_translation(node.op.type)
            if translation.has_indexed_method(method_name):
                translation.apply_method(method_name, node, graph, *args)

    def apply_total(self, method_name, graph, *args):
        for node in graph.list_nodes():
            self.apply_specific(node.op.type, method_name, node, graph, *args)

    def register(self, translation, *op_types):
        for op_type in op_types:
            if op_type in self.translations:
                raise KeyError("A translation is already registed for op type '%s'" % op_type)
            self.translations[op_type] = translation

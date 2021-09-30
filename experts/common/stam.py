class global_config(property):
    def __init__(self, fget=..., fset=..., fdel=..., doc=...) -> None:
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)
        self.__value = 2

    def getter(self, fget):
        return self.__value

    def setter(self, fset):
        self.__value = super().setter(fset)
        return self.__value


class foo:
    @global_config
    def bar(self):
        pass

    @bar.setter
    def bar(self, v):
        return int(v)


from dataclasses import dataclass

__all__ = ("Wrapper",)


@dataclass
class Wrapper:
    def __init__(self, wrapped, setup_func=None, omitted_member=None, **others):
        self.__wrapped__ = wrapped
        self.__others__ = others
        omitted_member = [] if omitted_member is None else omitted_member
        if setup_func is not None:
            setup_func(self.__wrapped__, self.__others__)
        for m in dir(self.__wrapped__):
            if (
                m[:2] != "__"
                and m not in dir(self.__class__)
                and m not in omitted_member
            ):
                self.__dict__[m] = getattr(self.__wrapped__, m)

    @property
    def wrap(self):
        return self.__wrapped__

    @property
    def others(self):
        return self.__others__

    def other(self, key):
        return self.__others__[key]

    def __repr__(self):
        return f"Wrapper({repr(self.wrap)})"

    def __getattr__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        elif key in self.__others__:
            return self.__others__[key]
        raise AttributeError(
            f"`{self.__class__.__name__}` object wrapping `{self.wrap}` has no attribute `{key}`."
        )

import pathlib


class CustomPath(pathlib.Path):
    """A custom path class that can be formatted to display as a hyperlink in terminal."""

    # This is a hack to inherit pathlib.Path and initialize the _flavour property.
    # https://stackoverflow.com/questions/61689391/error-with-simple-subclassing-of-pathlib-path-no-flavour-attribute
    # noinspection PyProtectedMember
    # noinspection PyUnresolvedReferences
    _flavour = type(pathlib.Path())._flavour

    def __format__(self, format_spec):
        if format_spec == '':
            return str(self)
        elif format_spec == 'link':
            return _create_hyperlink(self.absolute())
        elif format_spec == 'exists':
            if self.exists():
                return _create_hyperlink(self.absolute())
            else:
                return _create_hyperlink(self.absolute()) + ' does not exist. \nParent directory: ' + _create_hyperlink(
                    self.parent.absolute())
        else:
            raise NotImplementedError(f"Format spec {format_spec} is not implemented")

    def __iadd__(self, other: str):
        return CustomPath(str(self) + other)


def _create_hyperlink(text: [str, pathlib.Path]):
    if isinstance(text, pathlib.Path):
        return f'file://' + str(text)
    else:
        return f'file://' + text


if __name__ == '__main__':
    path = CustomPath("/18t1/osmosis_utils/data/RGBD/scan_net_pp/dataset/data")
    print(f"{path:exists}")
    path = path / "./train/r_0"
    path += ".png"
    print(f"{path:exists}")

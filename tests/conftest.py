import pytest
import yaml
import hyperprophet
import pandas as pd

TEST_GLOBALS = {
    'Prophet': hyperprophet.Prophet,
    'hyperprophet': hyperprophet
}

def pytest_collect_file(parent, path):
    if path.ext == ".yml" and path.basename.startswith("test"):
        return YAMLFile.from_parent(parent, fspath=path)

class YAMLFile(pytest.File):
    def collect(self):
        items = yaml.safe_load_all(self.fspath.open())
        for spec in items:
            yield YAMLItem.from_parent(self, name=spec['name'], spec=spec)

class YAMLItem(pytest.Item):
    def __init__(self, name, parent, spec):
        super().__init__(name, parent)
        self.spec = spec
        self.vars = {}

    def runtest(self):
        self.load_vars()
        exec(self.spec['test'], self.vars)
        self.verify(self.vars.get("result"), self.vars.get('expected_result'))

    def load_vars(self):
        self.vars = dict(TEST_GLOBALS)
        self.vars.update({k: compile_data(v) for k, v in self.spec['vars'].items()})

    def verify(self, result, expected):
        if expected is not None and type(expected) in VALIDATORS:
            v = VALIDATORS[type(expected)]
            v.validate(result, expected)
        else:
            assert result == expected

    def reportinfo(self):
        return self.fspath, 0, "test: {}".format(self.name)

def compile_data(data):
    """Compiles the data to convert the data to the required types when
    $type is specified.

            >>> compile({"$type": "set", "data": [1, 2, 1]})
            {1, 2}

        See register_type for more details.
    """
    if isinstance(data, dict):
        if "$type" in data:
            type_ = data.pop('$type')
            if type_ not in ADAPTERS:
                raise ValueError(f"Invalid type: {type_}")
            adapter = ADAPTERS[type_]
        else:
            adapter = None

        data = {k: compile_data(v) for k, v in data.items()}

        if adapter:
            return adapter.process(data)
        else:
            return data
    elif isinstance(data, list):
        return [compile_data(d) for d in data]
    else:
        return data

ADAPTERS = {}
VALIDATORS = {}

def register_adapter(name, adapter):
    ADAPTERS[name] = adapter
    VALIDATORS[adapter.type] = adapter

class BaseAdapter:
    def __init__(self, type, hints=None):
        self.type = type
        self.hints = hints or {}

    def validate(self, result, expected):
        assert result == expected

class SetAdapter(BaseAdapter):
    def process(self, args):
        return set(args['data'])

class DataFrameAdapter(BaseAdapter):
    def process(self, args):
        return pd.DataFrame(args['data'], columns=args['columns'])

    def validate(self, result, expected):
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Expected dataframe, found {}".format(type(result)))
        d1 = result.sort_index().to_dict(orient='split')
        d2 = expected.sort_index().to_dict(orient='split')
        assert d1['columns'] == d2['columns']
        assert d1['data'] == d2['data']

register_adapter('DataFrame', DataFrameAdapter(pd.DataFrame))
register_adapter('Set', SetAdapter(set))

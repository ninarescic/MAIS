import configparser
import os
import io

from sklearn.model_selection import ParameterGrid

def string_to_value(s):
    """
    If string is convertable to int, returns int,
    if it is convertable to float, returns float,
    if it is comma separated, returns list of strings,
    otherwise returns original string.
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            pass

    if "," in s:
        list_of_values = s.split(",")
        return [val.strip() for val in list_of_values]
    else:
        return s



class ConfigFile():

    """
    Class encapsulating the ConfigParser. 
    Deals with INI files.
    """

    def __init__(self, param_dict=None):

        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        if param_dict:
            for name, value in param_dict.items():
                self.config[name] = value

    def save(self, filename):
        if type(filename) == str:
            with open(filename, 'w', encoding="utf-8") as configfile:
                self.config.write(configfile)
        else:
            self.config.write(filename)

    def to_string(self):
        output = io.StringIO()
        self.config.write(output)
        ret = output.getvalue()
        output.close()
        return ret

    def load(self, filename):
        if not os.path.exists(filename):
            raise ValueError(f"Config file {filename} not exists. Provide name (including path) to a valid config file.")
        self.config.read(filename)

    def section_as_dict(self, section_name):
        sdict = self.config._sections.get(section_name, {})
        return {name: string_to_value(value) for name, value in sdict.items()}

    def fix_output_id(self):
        output_id = self.section_as_dict("OUTPUT_ID").get("id", None)
        if output_id is None:
            return 
        text_id = ""
        for variable in output_id:
            section, name = variable.split(":")
            text_id += f"_{section}_{name}={self.section_as_dict(section).get(name, None)}"
        text_id = text_id.replace(" ", "_")
        self.config["OUTPUT_ID"]["id"] = text_id



class ConfigFileGenerator():
    """
    Class to generate ConfigFiles from a config file with lists of parameters. 
    """
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str # this is to keep the case of the keys in the config file

    def _explode_lists(self, section):
        return {
            name : value.split(";")
            for name, value in section.items()
        }

    def load(self, filename):
        if not os.path.exists(filename):
            raise ValueError(f"Config file {filename} not exists. Provide name (including path) to a valid config file.")
        self.config.read(filename)


        variable_names = {
            section: list(self.config._sections[section].keys())
            for section in self.config.sections()
        }   

        # convert to dict 
        param_dict = {
            section: self._explode_lists(self.config._sections[section])
            for section in self.config.sections()
        }
        # convert each section to the list of final sections
        param_dict = {
            name: list(ParameterGrid(section))
            for name, section in param_dict.items()
        }
        # do the outer parameter grid
        param_dict = ParameterGrid(param_dict)
        for params in param_dict:
            x = ConfigFile(param_dict=params)
            x.fix_output_id()
            yield x


if __name__ == "__main__":

    test_generator = ConfigFileGenerator()
    for config in test_generator.load("../../config/info_verona.ini"):
        print(config.section_as_dict("OUTPUT_ID").get("id", None))
        print(config.to_string())


    exit()
    test_dict = {
        "TASK": {"num_nodes": 10000},
        "MODEL": {"beta": 0.155,
                  "gamma": 1/12.39,
                  "sigma": 1/5.2
                  }
    }

    test_config = ConfigFile(test_dict)
    test_config.save("test.ini")

    new_config = ConfigFile()
    new_config.load("test.ini")

    print(new_config.section_as_dict("TASK"))
    print(new_config.section_as_dict("MODEL"))

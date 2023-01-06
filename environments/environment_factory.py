
class MaGridworldFactory(object):

    def get_environment(self, environment_settings):
        from environments.ma_gridworld import MAGridworld
        return MAGridworld(**environment_settings)

class SysAdminFactory(object):

    def get_environment(self, environment_settings):
        from environments.sys_admin import SysAdmin
        return SysAdmin(**environment_settings)

environment_factories = {
    'ma_gridworld' : MaGridworldFactory,
    'sys_admin' : SysAdminFactory,
}

def get_environment(environment_type, environment_settings):
    return environment_factories[environment_type]().get_environment(environment_settings)
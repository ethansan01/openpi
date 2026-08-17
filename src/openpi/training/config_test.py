from openpi.training import config as _config


def test_pi05_libero_relative_eef6d_no_state_config():
    config = _config.get_config("pi05_libero_relative_eef6d_no_state_low_mem_finetune")

    assert config.model.pi05
    assert config.model.action_horizon == 10
    assert config.model.action_dim == 32
    assert config.model.discrete_state_input is False
    assert config.data.repo_id == "local/libero_relative_eef6d"
    assert config.data.relative_eef6d_actions
    assert config.ema_decay is None

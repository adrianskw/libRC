import numpy as np

from libRC import mapRC, diffRC


def synthetic_signal(D=2, M=300, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20 * np.pi, M)
    y = np.vstack([np.sin(t + i) for i in range(D)])
    y += 0.01 * rng.standard_normal(y.shape)
    return y


class TestMapRCPipeline:
    def test_listen_train_echo_infer_roundtrip(self):
        np.random.seed(0)
        N, D, M = 60, 2, 300
        y = synthetic_signal(D=D, M=M)

        RC = mapRC(N, bias=True)
        RC.makeConnectionMatDegree(rho=0.9, degree=5)
        RC.makeInputMat(D=D, sigma=0.5)

        RC.listen(y)
        assert RC.r.shape == (N, M)
        assert np.all(np.isfinite(RC.r))

        RC.train(y, alpha=0.01)
        assert RC.W.shape == (D, N + 1)  # +1 for bias row
        assert RC.y_est.shape == (D, M)
        assert np.all(np.isfinite(RC.y_est))
        assert RC.fitError < 1.0  # sanity bound on a smooth, low-noise signal

        M_echo = 50
        RC.echo(M_echo)
        assert RC.y_echo.shape == (D, M_echo)
        assert np.all(np.isfinite(RC.y_echo))

        driveIndex = [0]
        y_drive = synthetic_signal(D=D, M=50, seed=1)[driveIndex]
        RC.infer(y_drive, driveIndex)
        assert RC.y_infer.shape == (D, 50)
        assert np.all(np.isfinite(RC.y_infer))

    def test_train_is_idempotent_wrt_bias_row(self):
        np.random.seed(1)
        N, D, M = 30, 1, 100
        y = synthetic_signal(D=D, M=M)
        RC = mapRC(N, bias=True)
        RC.makeConnectionMatDegree(rho=0.8, degree=3)
        RC.makeInputMat(D=D, sigma=0.5)
        RC.listen(y)
        RC.train(y, alpha=0.01)
        RC.train(y, alpha=0.02)
        assert RC.r.shape == (N + 1, M)

    def test_explicit_mask_does_not_raise(self):
        # regression test for issue #1: `if mask == None` crashed on real arrays
        np.random.seed(5)
        N, D, M = 20, 2, 100
        y = synthetic_signal(D=D, M=M)
        RC = mapRC(N)
        RC.makeConnectionMatDegree(rho=0.8, degree=3)
        RC.makeInputMat(D=D, sigma=0.5)
        RC.listen(y)
        RC.train(y, alpha=0.01, mask=np.eye(D))
        assert np.all(np.isfinite(RC.y_est))


class TestDiffRCPipeline:
    def test_listen_train_echo_roundtrip(self):
        np.random.seed(2)
        N, D, M = 40, 2, 300
        y = synthetic_signal(D=D, M=M)

        RC = diffRC(N, ds=0.1)
        RC.chooseIntegrator("RK4")
        RC.makeConnectionMatDegree(rho=0.9, degree=4)
        RC.makeInputMat(D=D, sigma=0.5)

        RC.listen(y)
        RC.train(y, alpha=0.01)
        RC.echo(30)

        assert RC.y_echo.shape == (D, 30)
        assert np.all(np.isfinite(RC.y_echo))

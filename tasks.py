from invoke import task

SRC_LOCATIONS = "qml examples tasks.py"
TEST_LOCATIONS = "qml"
EXAMPLE_LOCATIONS = "examples"


@task
def mypy(c):
    c.run(f"mypy {SRC_LOCATIONS}")


@task
def test(c, pdb=False):
    args = ""
    if pdb:
        args += " --pdb"
    c.run(f"pytest --failed-first {args} {SRC_LOCATIONS}", pty=True)


@task
def testm(c, pdb=False):
    args = ""
    if pdb:
        args += " --pdb"
    c.run(f"pytest --testmon {args} {SRC_LOCATIONS}", pty=True)


@task
def black(c):
    c.run(f"black {SRC_LOCATIONS} {TEST_LOCATIONS}")


@task
def black_check(c):
    c.run(f"black --check {SRC_LOCATIONS} {TEST_LOCATIONS}")


@task
def autoflake(c):
    c.run(
        f"autoflake --recursive --in-place --remove-all-unused-imports --remove-unused-variables {SRC_LOCATIONS} {TEST_LOCATIONS}"
    )


@task
def autoflake_check(c):
    c.run(
        f"autoflake --recursive --check --remove-all-unused-imports --remove-unused-variables {SRC_LOCATIONS} {TEST_LOCATIONS}"
    )


@task
def isort(c):
    c.run(f"isort {SRC_LOCATIONS} {TEST_LOCATIONS}")


@task
def isort_check(c):
    c.run(f"isort --check {SRC_LOCATIONS} {TEST_LOCATIONS}")


# @task
# def lint_imports(c):
#     c.run("lint-imports")


@task
def fix(c):
    autoflake(c)
    isort(c)
    black(c)


@task
def check(c):
    autoflake_check(c)
    isort_check(c)
    black_check(c)
    # lint_imports(c)

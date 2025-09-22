from pathlib import Path
from sqlalchemy import text, create_engine


def get_fitness_over_generations(population_count, db_path):
    # Create engine for SQLite database
    engine = create_engine(f"sqlite:///{db_path}")

    all_fitnesses = []
    buffer = []

    # Query raw SQL
    with engine.connect() as conn:
        result = conn.execute(text("SELECT fitness_ FROM individual"))
        for row in result:
            buffer.append(row[0])  # row is a tuple, so take first element
            if len(buffer) == population_count:
                all_fitnesses.append(buffer)
                buffer = []

    # Add remainder if not divisible by 10
    if buffer:
        all_fitnesses.append(buffer)

    return all_fitnesses


# print(get_fitness_over_generations(10, f"{Path.cwd()}/__data__/database.db"))





from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfigAG:

    nombre: str

    # Población
    sol_per_pop: int = 100
    num_generations: int = 500

    # Selección
    parent_selection_type: str = "tournament"
    K_tournament: int = 3
    num_parents_mating: int = 50

    # Cruce
    crossover_type: str = "two_points"

    # Mutación
    mutation_type: str = "random"
    mutation_probability: float = 0.05

    # Elitismo
    keep_elitism: int = 5

    # Criterios de parada
    stop_criteria: list = field(
        default_factory=lambda: ["reach_53.25", "saturate_50"]
    )

    # Genes (fijos para este problema)
    num_genes: int = 50
    gene_type: type = int
    gene_space: Optional[object] = None

    def __post_init__(self):
        if self.gene_space is None:
            self.gene_space = range(0, 20)

    def a_dict_pygad(self) -> dict:
        return {
            "sol_per_pop":          self.sol_per_pop,
            "num_genes":            self.num_genes,
            "gene_type":            self.gene_type,
            "gene_space":           self.gene_space,
            "parent_selection_type": self.parent_selection_type,
            "K_tournament":         self.K_tournament,
            "keep_elitism":         self.keep_elitism,
            "crossover_type":       self.crossover_type,
            "num_parents_mating":   self.num_parents_mating,
            "mutation_type":        self.mutation_type,
            "mutation_probability": self.mutation_probability,
            "num_generations":      self.num_generations,
            "stop_criteria":        self.stop_criteria,
        }

    def resumen(self) -> str:
        return (
            f"{self.nombre} | pop={self.sol_per_pop} | "
            f"mut={self.mutation_probability} | elite={self.keep_elitism} | "
            f"cx={self.crossover_type} | sel={self.parent_selection_type} | "
            f"K={self.K_tournament}"
        )


CONFIGS_POBLACION = [
    # Solo varía sol_per_pop; num_parents_mating se fija al valor base (50)
    # para comparar una única variable a la vez (criterio de comparación justa).
    ConfigAG("Pop_050", sol_per_pop=50),
    ConfigAG("Pop_100", sol_per_pop=100),
    ConfigAG("Pop_200", sol_per_pop=200),
]

CONFIGS_MUTACION = [
    ConfigAG("Mut_001", mutation_probability=0.01),
    ConfigAG("Mut_005", mutation_probability=0.05),
    ConfigAG("Mut_015", mutation_probability=0.15),
    ConfigAG("Mut_030", mutation_probability=0.30),
]

CONFIGS_CRUCE = [
    ConfigAG("Cx_SinglePoint", crossover_type="single_point"),
    ConfigAG("Cx_TwoPoints",   crossover_type="two_points"),
    ConfigAG("Cx_Uniform",     crossover_type="uniform"),
]

CONFIGS_SELECCION = [
    ConfigAG("Sel_Tournament_K3",  parent_selection_type="tournament", K_tournament=3),
    ConfigAG("Sel_Tournament_K7",  parent_selection_type="tournament", K_tournament=7),
    ConfigAG("Sel_RWS",            parent_selection_type="rws"),
    ConfigAG("Sel_SUS",            parent_selection_type="sus"),
    ConfigAG("Sel_Rank",           parent_selection_type="rank"),
]

CONFIGS_ELITISMO = [
    ConfigAG("Elite_0",  keep_elitism=0),
    ConfigAG("Elite_5",  keep_elitism=5),
    ConfigAG("Elite_15", keep_elitism=15),
]

CONFIG_BASE = ConfigAG("Base")

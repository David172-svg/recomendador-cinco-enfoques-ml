import unittest

from src.datos import USUARIOS, cargar_catalogo, generar_interacciones, peliculas_no_vistas
from src.metodos import METODOS


class PruebasProyecto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogo = cargar_catalogo()
        cls.interacciones = generar_interacciones(cls.catalogo)

    def test_catalogo_y_reproducibilidad(self):
        self.assertEqual(len(self.catalogo), 32)
        self.assertEqual(len(self.interacciones), len(USUARIOS) * 17)
        self.assertEqual(self.interacciones, generar_interacciones(self.catalogo))

    def test_cinco_metodos_recomiendan_solo_peliculas_no_vistas(self):
        self.assertEqual(len(METODOS), 5)
        for usuario in USUARIOS:
            permitidas = {p["titulo"] for p in peliculas_no_vistas(
                self.catalogo, self.interacciones, usuario)}
            for clave, metodo in METODOS.items():
                with self.subTest(usuario=usuario, metodo=clave):
                    resultado = metodo(self.catalogo, self.interacciones, usuario)
                    self.assertTrue(resultado["recomendaciones"])
                    self.assertTrue(resultado["algoritmo"])
                    for titulo, _ in resultado["recomendaciones"]:
                        self.assertIn(titulo, permitidas)


if __name__ == "__main__":
    unittest.main()

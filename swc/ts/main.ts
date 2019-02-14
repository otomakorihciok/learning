enum Color {
  RED,
  GRREN,
  BLUE,
}

interface Pallet {
  colors: Color[];
}

const p = <Pallet>{
  colors: [Color.RED, Color.GRREN],
};

p.colors.forEach(c => console.log(c));

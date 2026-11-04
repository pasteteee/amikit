// Usage: node make_cheek.mjs <out.json> -- Shorts v2: penguin chick cheek (pink), MR + sc + inc
import { buildShort } from './engine/short.mjs';
import fs from 'fs';
const scn = buildShort({
  rounds: ['mr 6', 'inc 6 to 12'],
  title: 'Щёчка пингвинёнка',
  sub: 'крючком за минуту · кольцо, сбн, прибавка',
  outroHead: 'Щёчка готова!',
  outroSub: 'Свяжите вторую — и пришейте пингвинёнку',
  outroText: 'Оставьте хвост нити, чтобы пришить.',
});
scn.meta.render = {
  colour: '#f77ea0', sheen: '#ffd0dc', hookColour: '#7fd6c8', handleColour: '#4b2a5e', exposure: 0.95,
  backdrop: [[0, '#fbf3ea'], [0.5, '#f4e4d4'], [1, '#e6cbb4']],
  toyImage: '/web/assets/penguin.jpg',
};
fs.writeFileSync(process.argv[2], JSON.stringify(scn));

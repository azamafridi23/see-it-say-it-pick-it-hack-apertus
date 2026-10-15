import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { stripTypeScriptTypes } from "node:module";

// Run the actual state engine, transpiled locally; no server or model involved.
const source = readFileSync(new URL("../src/lib/simulation.ts", import.meta.url), "utf8");
const outputText = stripTypeScriptTypes(source);
const { frames, initialFrame, initialTopFrame, TOP_TRAYS, elbow, motion, robotMotion3D, robotWorld, rotateTravel } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const fixture = actions => ({
  objects: [{ id: "red_cube", type: "red cube", pixel: [200, 400] }, { id: "yellow_gear", type: "yellow gear", pixel: [400, 500] }, { id: "black_tray", type: "black tray" }],
  image_size: [1000, 800], answer: { actions },
});

test("pick-and-hold keeps the selected object in the gripper", () => {
  const result = fixture([{ action: "pick", object: "red_cube" }]);
  const states = frames(result);
  assert.equal(states[0].held, null);
  assert.equal(states[1].held, "red_cube");
  assert.equal(states[1].positions.red_cube.y, states[1].gripper.y + 34);
  assert.deepEqual(states[0], initialFrame(result));
});
test("place refers to the destination and releases the previously picked object", () => {
  const states = frames(fixture([{ action: "pick", object: "red_cube" }, { action: "place", object: "black_tray" }]));
  assert.equal(states[2].held, null);
  assert.ok(states[2].positions.red_cube.x >= 635 && states[2].positions.red_cube.x <= 778);
  assert.equal(states[2].positions.red_cube.y, 350);
  assert.equal(states[2].positions.black_tray.x, 705);
});
test("place onto an object stacks above the support", () => {
  const states = frames(fixture([{ action: "pick", object: "red_cube" }, { action: "place", object: "yellow_gear" }]));
  assert.equal(states[2].positions.red_cube.x, states[2].positions.yellow_gear.x);
  assert.equal(states[2].positions.red_cube.y, states[2].positions.yellow_gear.y - 34);
});
test("impossible successive picks stop playback instead of discarding the held object", () => {
  const states = frames(fixture([{ action: "pick", object: "red_cube" }, { action: "pick", object: "yellow_gear" }, { action: "place", object: "black_tray" }]));
  assert.equal(states.length, 3);
  assert.equal(states[2].held, "red_cube");
  assert.ok(states[2].error);
});
test("ask and refuse never move objects", () => {
  for (const action of ["ask_user", "refuse"]) {
    const states = frames(fixture([{ action, question: "Which cube?", reason: "Unsafe" }]));
    assert.deepEqual(states[0], states[1]);
  }
});
test("every saved plan can be replayed without missing-object or gripper errors", () => {
  for (const filename of ["path_a_dev.json", "path_b_dev.json"]) {
    const saved = JSON.parse(readFileSync(new URL(`../../../results/${filename}`, import.meta.url), "utf8"));
    for (const answer of saved.results) {
      const ids = [...new Set(answer.actions.map(a => a.object).filter(Boolean))];
      const states = frames({ answer, objects: ids.map((id, i) => ({ id, type: id, pixel: [100 + i * 80, 300] })), image_size: [1000, 800] });
      assert.ok(states.every(state => !state.error), answer.instruction_id);
    }
  }
});
test("inverse kinematics reaches targets throughout the illustrated workspace", () => {
  const base = { x: 585, y: 336 };
  for (const target of [{ x: 85, y: 235 }, { x: 750, y: 316 }, { x: 200, y: 316 }]) {
    const joint = elbow(target);
    assert.ok(Math.abs(Math.hypot(joint.x - base.x, joint.y - base.y) - 280) < 1e-6);
    assert.ok(Math.abs(Math.hypot(joint.x - target.x, joint.y - target.y) - 280) < 1e-6);
  }
});

test("top and side views honor black tray membership", () => {
  const result = fixture([]);
  result.objects[0].in = "black_tray";
  result.objects[1].in = "grey_tray";
  const frame = initialTopFrame(result);
  const black = TOP_TRAYS.black_tray, grey = TOP_TRAYS.grey_tray;
  assert.ok(frame.positions.red_cube.x > black.x && frame.positions.red_cube.x < black.x + black.width);
  assert.ok(frame.positions.yellow_gear.x > grey.x && frame.positions.yellow_gear.x < grey.x + grey.width);
  assert.ok(initialFrame(result).positions.red_cube.x > 635);
});
test("unknown replay membership stays outside labeled trays", () => {
  const frame = initialTopFrame(fixture([]));
  assert.ok(frame.positions.red_cube.y > 320);
});
test("top view placement releases objects into destination tray with synchronized action count", () => {
  const result = fixture([{ action: "pick", object: "red_cube" }, { action: "place", object: "black_tray" }]);
  const top = frames(result, "top"), side = frames(result);
  assert.equal(top.length, side.length);
  assert.equal(top[1].held, "red_cube");
  assert.equal(top[2].held, null);
  assert.ok(top[2].positions.red_cube.x > TOP_TRAYS.black_tray.x);
  assert.ok(top[2].positions.red_cube.y < 320);
});


test("grip and release occur at contact before the arm lifts", () => {
  const states = frames(fixture([{ action: "pick", object: "red_cube" }, { action: "place", object: "black_tray" }]));
  const pick = motion(states[0], states[1], .65);
  assert.equal(pick.held, "red_cube");
  assert.equal(pick.target.y + 34, states[0].positions.red_cube.y);
  assert.equal(motion(states[0], states[1], .5).held, null);
  const drop = motion(states[1], states[2], .65);
  assert.equal(drop.held, null);
  assert.equal(drop.target.y + 34, states[2].positions.red_cube.y);
  assert.deepEqual(motion(states[1], states[2], 1).target, states[2].gripper);
});

test("elbow stays above the pedestal when reaching either tray", () => {
  for (const x of [85, 200, 705, 750]) assert.ok(elbow({ x, y: 316 }).y < 336);
});


test("3D tray transfer sweeps around the pedestal and keeps its load", () => {
  const states = frames(fixture([{ action: "pick", object: "red_cube" }, { action: "place", object: "black_tray" }]));
  const start = robotMotion3D(states[1], states[2], 0);
  const middle = robotMotion3D(states[1], states[2], .175);
  const arrived = robotMotion3D(states[1], states[2], .35);
  assert.ok(middle.target.y > 100, "arm should sweep into depth, not travel through the base");
  assert.ok(middle.yaw < start.yaw && middle.yaw > arrived.yaw);
  assert.equal(middle.held, "red_cube");
  assert.deepEqual(robotMotion3D(states[1], states[2], 1).target, robotWorld(states[2].gripper));
  assert.equal(robotMotion3D(states[1], states[2], .65).held, null);
  for (let i = 0; i <= 100; i++) {
    const pose = robotMotion3D(states[1], states[2], i / 100);
    assert.ok(Math.hypot(pose.target.x, pose.target.y) >= 35 - 1e-6);
  }
});

test("base rotation follows the short arc across the 360 degree seam", () => {
  const point = degrees => ({ x: 100 * Math.cos(degrees * Math.PI / 180), y: 100 * Math.sin(degrees * Math.PI / 180), z: 80 });
  const middle = rotateTravel(point(350), point(10), .5);
  assert.ok(Math.abs(middle.x - 100) < 1e-6);
  assert.ok(Math.abs(middle.y) < 1e-6);
  const back = rotateTravel(point(10), point(350), .5);
  assert.ok(Math.abs(back.x - 100) < 1e-6);
});

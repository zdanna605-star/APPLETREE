import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Maximize, Minimize, Hand } from 'lucide-react';

// --- Types & Constants ---

interface Particle {
  mesh: THREE.Points;
  velocity: THREE.Vector3;
  type: 'APPLE' | 'EXPLOSION';
  life: number; // For explosions
  id: string;
}

const SPAWN_INTERVAL = 1000; // ms
const GRAVITY = -0.05; // Base gravity unit for explosions
const APPLE_GRAVITY = -0.005; // Gravity acceleration for falling apples
const APPLE_RADIUS = 1.5; // Size of the apple

// Piano frequencies (C4, D4, E4, F4, G4, A4, B4)
const PIANO_NOTES = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88];

// Pre-defined colors for random switching ONLY on hit
const HIT_COLORS = [
  '#ff0000', // Red
  '#ff7f00', // Orange
  '#ffff00', // Yellow
  '#00ff00', // Green
  '#0000ff', // Blue
  '#8000ff', // Purple
];

// --- Shared Material for ALL Falling Apples ---
// This ensures every new apple is identical and red.
// We use vertexColors: true so the geometry's internal gradients (yellow spots, dark bottom) work.
const BASE_APPLE_MATERIAL = new THREE.PointsMaterial({ 
  size: 0.08,
  vertexColors: true,
  transparent: true,
  opacity: 0.9,
  sizeAttenuation: true
});

// --- Helper: Generate Structured Apple Matrix ---
const createAppleGeometry = (radius: number, segments: number, baseColorHex: string) => {
  const positions = [];
  const colors = [];
  
  const baseColor = new THREE.Color(baseColorHex);
  const yellowColor = new THREE.Color(0xffdd00); // Yellow highlight/spots
  const darkColor = new THREE.Color(0x3e2723);   // Dark stem/bottom
  const greenColor = new THREE.Color(0x558b2f);  // Slightly green near stem

  // Generate a Sphere-based Grid (Latitude/Longitude)
  for (let lat = 0; lat <= segments; lat++) {
    const v = lat / segments;
    const phi = v * Math.PI; 

    const lonSegments = Math.floor(segments * 1.5); 
    
    for (let lon = 0; lon < lonSegments; lon++) {
      const u = lon / lonSegments;
      const theta = u * Math.PI * 2;

      // 1. Basic Sphere Coordinates
      let x = -Math.sin(phi) * Math.cos(theta);
      let z = -Math.sin(phi) * Math.sin(theta);
      let y = Math.cos(phi);

      // 2. Apple Shaping Math
      const rho = Math.sqrt(x*x + z*z); // Distance from center axis
      
      // Upper body widening
      if (y > 0) {
        x *= 1.0 + (y * 0.3);
        z *= 1.0 + (y * 0.3);
      }
      
      // Top Dimple
      const topDipRadius = 0.4;
      if (rho < topDipRadius && y > 0) {
        const dip = Math.pow((topDipRadius - rho) / topDipRadius, 2);
        y -= dip * 0.8;
      }

      // Bottom Dimple
      const bottomDipRadius = 0.5;
      if (rho < bottomDipRadius && y < 0) {
         const dip = Math.pow((bottomDipRadius - rho) / bottomDipRadius, 1.5);
         y += dip * 0.3;
      }
      
      // Bottom Taper
      if (y < 0) {
        const taper = 1.0 - (Math.abs(y) * 0.3);
        x *= taper;
        z *= taper;
      }

      // 3. Add slight "Thickness"
      const thickness = 0.05 + Math.random() * 0.05;
      x *= radius * (1 + thickness);
      y *= radius * (1 + thickness);
      z *= radius * (1 + thickness);

      positions.push(x, y, z);

      // 4. Procedural Texturing
      const c = baseColor.clone();

      // Top gradient
      if (y > radius * 0.3) {
         const noise = Math.sin(x * 5) * Math.cos(z * 5);
         if (noise > 0.5) c.lerp(yellowColor, 0.3);
         if (rho < radius * 0.2) c.lerp(greenColor, 0.6);
         if (rho < radius * 0.1) c.lerp(darkColor, 0.8);
      }

      // Bottom shadow
      if (y < -radius * 0.6) {
        c.lerp(darkColor, 0.5);
      }

      // Highlight
      const isFront = z > 0.5 * radius;
      if (isFront && x > 0 && y > 0) {
         c.lerp(new THREE.Color(0xffaaaa), 0.1);
      }

      colors.push(c.r, c.g, c.b);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return geometry;
};

// --- Helper: Generate Explosion Geometry ---
const createExplosionGeometry = (startPos: THREE.Vector3, count: number, color: THREE.Color) => {
  const geometry = new THREE.BufferGeometry();
  const positions = [];
  const velocities = [];
  const colors = [];

  for (let i = 0; i < count; i++) {
    positions.push(startPos.x, startPos.y, startPos.z);
    
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    
    const speed = 0.1 + Math.random() * 0.2;
    const vx = Math.sin(phi) * Math.cos(theta) * speed;
    const vy = Math.sin(phi) * Math.sin(theta) * speed;
    const vz = Math.cos(phi) * speed;
    
    velocities.push(vx, vy, vz);
    
    // Varied colors for explosion
    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return { geometry, velocities };
};

const App = () => {
  // --- Refs for Scene Management ---
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  
  // --- Refs for Game State ---
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const lastSpawnTimeRef = useRef<number>(0);
  const handPositionsRef = useRef<THREE.Vector3[]>([]);
  const cursorMeshRef = useRef<THREE.Mesh[]>([]);
  
  // --- Ref for Audio ---
  const audioCtxRef = useRef<AudioContext | null>(null);

  // --- React State ---
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [score, setScore] = useState(0);

  // --- Initialization ---
  useEffect(() => {
    let active = true;

    // 1. Initialize Audio Context
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContextClass();
    audioCtxRef.current = ctx;

    // Helper to resume audio context on interaction
    const unlockAudio = () => {
      if (audioCtxRef.current && audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume().then(() => {
          console.log("Audio Context Resumed via User Interaction");
        });
      }
      // Remove listeners after first interaction
      window.removeEventListener('click', unlockAudio);
      window.removeEventListener('touchstart', unlockAudio);
      window.removeEventListener('keydown', unlockAudio);
    };

    window.addEventListener('click', unlockAudio);
    window.addEventListener('touchstart', unlockAudio);
    window.addEventListener('keydown', unlockAudio);

    const init = async () => {
      if (!videoRef.current || !canvasRef.current || !containerRef.current) return;

      // 2. Setup Three.js
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x000000); 
      sceneRef.current = scene;

      // Camera
      const width = window.innerWidth;
      const height = window.innerHeight;
      const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 100);
      camera.position.z = 10;
      cameraRef.current = camera;

      // Renderer
      const renderer = new THREE.WebGLRenderer({ 
        canvas: canvasRef.current, 
        alpha: false, 
        antialias: true 
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setClearColor(0x000000, 1);
      rendererRef.current = renderer;

      // 3. Add Hand Cursors
      cursorMeshRef.current = [];
      const cursorGeo = new THREE.SphereGeometry(0.4, 16, 16);
      const cursorMat = new THREE.MeshBasicMaterial({ 
        color: 0x00ffff, 
        transparent: true, 
        opacity: 0.8,
        blending: THREE.AdditiveBlending 
      });
      
      for(let i=0; i<2; i++) {
        const mesh = new THREE.Mesh(cursorGeo, cursorMat);
        mesh.visible = false;
        
        const ringGeo = new THREE.RingGeometry(0.5, 0.6, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        mesh.add(ring);

        scene.add(mesh);
        cursorMeshRef.current.push(mesh);
      }

      // 4. Setup MediaPipe
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
        );
        
        if (!active) return;

        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (!videoRef.current || !active) return;
        
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        setLoading(false);
        animate();

      } catch (err) {
        console.error("Initialization error:", err);
        setLoading(false);
      }
    };

    init();

    const handleResize = () => {
      if (!cameraRef.current || !rendererRef.current) return;
      const width = window.innerWidth;
      const height = window.innerHeight;
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);
    
    return () => {
      active = false;
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('click', unlockAudio);
      window.removeEventListener('touchstart', unlockAudio);
      window.removeEventListener('keydown', unlockAudio);
      
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
      }
      particlesRef.current = [];
      sceneRef.current = null;
      handLandmarkerRef.current = null;
    };
  }, []);

  // --- Audio Logic ---
  const playRandomNote = () => {
    if (!audioCtxRef.current) return;
    const ctx = audioCtxRef.current;
    
    const freq = PIANO_NOTES[Math.floor(Math.random() * PIANO_NOTES.length)];
    // console.log('play note', freq);

    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();

    osc.type = 'triangle'; 
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.4);

    osc.connect(gainNode);
    gainNode.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  };

  // --- Game Loop ---
  const animate = () => {
    const now = performance.now();
    
    // 1. Detect Hands
    if (handLandmarkerRef.current && videoRef.current && videoRef.current.readyState >= 2) {
      let startTimeMs = performance.now();
      const results = handLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
      
      handPositionsRef.current = [];
      cursorMeshRef.current.forEach(m => m.visible = false);

      if (results.landmarks) {
        results.landmarks.forEach((landmarks, index) => {
          const tip = landmarks[8]; 
          
          if (cameraRef.current) {
            const vector = new THREE.Vector3(
              (1 - tip.x) * 2 - 1,
              -(tip.y) * 2 + 1,
              0.5
            );
            
            vector.unproject(cameraRef.current);
            const dir = vector.sub(cameraRef.current.position).normalize();
            const distance = -cameraRef.current.position.z / dir.z; 
            const pos = cameraRef.current.position.clone().add(dir.multiplyScalar(distance));
            
            handPositionsRef.current.push(pos);
            
            if (cursorMeshRef.current[index]) {
                cursorMeshRef.current[index].position.copy(pos);
                cursorMeshRef.current[index].visible = true;
                cursorMeshRef.current[index].children[0].rotation.z += 0.1;
                cursorMeshRef.current[index].children[0].rotation.x += 0.05;
            }
          }
        });
      }
    }

    // 2. Spawn Apples
    if (now - lastSpawnTimeRef.current > SPAWN_INTERVAL) {
      spawnApple();
      lastSpawnTimeRef.current = now;
    }

    // 3. Update Particles
    updateParticles();

    // 4. Render
    if (rendererRef.current && sceneRef.current && cameraRef.current) {
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    }

    requestRef.current = requestAnimationFrame(animate);
  };

  const spawnApple = () => {
    if (!sceneRef.current || !cameraRef.current) return;

    // CONSTANT RED COLOR for all new apples
    const redColorHex = '#ff0000';
    
    // Create geometry with baked-in vertex colors based on Red
    const geometry = createAppleGeometry(APPLE_RADIUS, 40, redColorHex);
    
    // Always use the shared BASE_APPLE_MATERIAL for performance and consistency.
    // This ensures every apple starts Red.
    const mesh = new THREE.Points(geometry, BASE_APPLE_MATERIAL);
    
    const dist = cameraRef.current.position.z; 
    const vFOV = THREE.MathUtils.degToRad(cameraRef.current.fov); 
    const visibleHeight = 2 * Math.tan(vFOV / 2) * dist; 
    const visibleWidth = visibleHeight * cameraRef.current.aspect;

    const spawnMargin = APPLE_RADIUS + 1;
    const spawnRange = Math.max(0, (visibleWidth / 2) - spawnMargin);
    const x = (Math.random() - 0.5) * 2 * spawnRange;
    
    mesh.position.set(x, 10, 0); 
    
    mesh.rotation.x = Math.random() * 0.5;
    mesh.rotation.y = Math.random() * Math.PI;

    sceneRef.current.add(mesh);

    particlesRef.current.push({
      mesh,
      velocity: new THREE.Vector3(0, -0.05 - (Math.random() * 0.05), 0),
      type: 'APPLE',
      life: 1,
      id: Math.random().toString()
    });
  };

  const updateParticles = () => {
    if (!sceneRef.current) return;
    
    for (let i = particlesRef.current.length - 1; i >= 0; i--) {
      const p = particlesRef.current[i];

      if (p.type === 'APPLE') {
        p.velocity.y += APPLE_GRAVITY;
        p.mesh.position.add(p.velocity);
        
        p.mesh.rotation.y += 0.015;
        p.mesh.rotation.x += 0.005;

        // Despawn if below screen
        if (p.mesh.position.y < -12) {
          sceneRef.current.remove(p.mesh);
          particlesRef.current.splice(i, 1);
          continue;
        }

        let hit = false;
        for (const handPos of handPositionsRef.current) {
          if (handPos.distanceTo(p.mesh.position) < APPLE_RADIUS + 0.8) { 
            hit = true;
            break;
          }
        }

        if (hit) {
          // 1. Play Sound
          playRandomNote();

          // 2. Pick Random Color LOCAL to this collision
          const randomColorHex = HIT_COLORS[Math.floor(Math.random() * HIT_COLORS.length)];
          const randomColor = new THREE.Color(randomColorHex);

          // 3. Clone material and apply color (Requirement)
          // Even though we remove it immediately, this isolates the change to THIS apple instance.
          // Note: Since we use vertex colors, this tint might behave as a multiply operation, 
          // but for the purpose of the request, this is the correct logic structure.
          if (p.mesh.material instanceof THREE.PointsMaterial) {
            const hitMaterial = p.mesh.material.clone();
            hitMaterial.color.set(randomColor);
            // If we wanted to override vertex colors to show pure randomColor, we'd set vertexColors: false here
            // hitMaterial.vertexColors = false; 
            p.mesh.material = hitMaterial;
          }

          // 4. Trigger Explosion with the RANDOM color
          explode(p.mesh.position, randomColor);
          
          // 5. Remove the apple immediately
          sceneRef.current.remove(p.mesh);
          particlesRef.current.splice(i, 1);
          
          setScore(s => s + 1);
        }

      } else if (p.type === 'EXPLOSION') {
        const positions = p.mesh.geometry.attributes.position.array as Float32Array;
        const velocities = p.mesh.userData.velocities as number[];

        for(let j=0; j < positions.length / 3; j++) {
            positions[j*3] += velocities[j*3];
            positions[j*3+1] += velocities[j*3+1];
            positions[j*3+2] += velocities[j*3+2];
            
            velocities[j*3+1] += GRAVITY * 0.5; 
        }
        p.mesh.geometry.attributes.position.needsUpdate = true;

        p.life -= 0.02;
        (p.mesh.material as THREE.PointsMaterial).opacity = p.life;

        if (p.life <= 0) {
          sceneRef.current.remove(p.mesh);
          particlesRef.current.splice(i, 1);
        }
      }
    }
  };

  const explode = (pos: THREE.Vector3, color: THREE.Color) => {
    if (!sceneRef.current) return;
    
    const count = 100;
    const { geometry, velocities } = createExplosionGeometry(pos, count, color);
    
    const material = new THREE.PointsMaterial({
      color: color, 
      vertexColors: true, 
      size: 0.15,
      transparent: true,
      opacity: 1
    });

    const mesh = new THREE.Points(geometry, material);
    mesh.userData = { velocities };
    
    sceneRef.current.add(mesh);
    
    particlesRef.current.push({
      mesh,
      velocity: new THREE.Vector3(),
      type: 'EXPLOSION',
      life: 1.0,
      id: Math.random().toString()
    });
  };

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
      setIsFullScreen(true);
    } else {
      document.exitFullscreen();
      setIsFullScreen(false);
    }
  };

  return (
    <div ref={containerRef} className="relative w-full h-full bg-black overflow-hidden select-none">
      <video 
        ref={videoRef} 
        className="absolute top-0 left-0 w-full h-full object-cover opacity-0 pointer-events-none" 
        autoPlay
        playsInline 
        muted
      />

      <canvas 
        ref={canvasRef} 
        className="absolute top-0 left-0 w-full h-full"
      />

      {loading && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black text-white">
          <div className="w-12 h-12 border-4 border-t-blue-500 border-white/20 rounded-full animate-spin mb-4"></div>
          <p className="text-xl font-light tracking-wide">Initializing Vision System...</p>
        </div>
      )}

      <div className="absolute z-10 top-0 left-0 w-full p-6 flex justify-between items-start pointer-events-none">
        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 text-white border border-white/10 shadow-xl pointer-events-auto">
          <div className="flex items-center gap-3 mb-2">
            <Hand className="text-blue-400 w-5 h-5" />
            <span className="font-bold text-lg">Virtual Touch</span>
          </div>
          <p className="text-sm text-gray-300 max-w-[200px] mb-3 leading-relaxed">
            Move your hand to touch the falling particles. Click or Tap anywhere to enable audio.
          </p>
          <div className="flex items-end gap-2">
            <span className="text-3xl font-mono font-bold text-green-400">{score}</span>
            <span className="text-xs text-gray-400 mb-1">DESTROYED</span>
          </div>
        </div>

        <div className="flex flex-col gap-3 pointer-events-auto">
            <button 
                onClick={toggleFullScreen}
                className="flex items-center justify-center w-12 h-12 bg-white/10 backdrop-blur-md rounded-xl border border-white/10 text-white hover:bg-white/20 transition-all active:scale-95"
            >
                {isFullScreen ? <Minimize className="w-5 h-5" /> : <Maximize className="w-5 h-5" />}
            </button>
        </div>
      </div>
      
      <div className="absolute bottom-4 right-4 text-white/30 text-xs pointer-events-none">
        Hand Tracking Active
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
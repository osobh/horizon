// Placeholder component for the node palette/toolbar
// This will contain draggable node types (GPU, CPU, Storage, etc.)

import React from 'react';

interface NodeType {
  id: string;
  name: string;
  icon: string;
  category: 'compute' | 'storage' | 'network';
}

const NODE_TYPES: NodeType[] = [
  { id: 'gpu', name: 'GPU Node', icon: '🎮', category: 'compute' },
  { id: 'cpu', name: 'CPU Node', icon: '💻', category: 'compute' },
  { id: 'storage', name: 'Storage Node', icon: '💾', category: 'storage' },
  { id: 'switch', name: 'Network Switch', icon: '🔀', category: 'network' },
];

export const NodePalette: React.FC = () => {
  const handleDragStart = (event: React.DragEvent, nodeType: NodeType) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(nodeType));
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div style={{
      width: '250px',
      height: '100%',
      backgroundColor: '#fff',
      borderRight: '1px solid #ddd',
      padding: '16px'
    }}>
      <h3>Node Types</h3>
      
      {['compute', 'storage', 'network'].map(category => (
        <div key={category} style={{ marginBottom: '20px' }}>
          <h4 style={{ 
            textTransform: 'capitalize',
            color: '#666',
            borderBottom: '1px solid #eee',
            paddingBottom: '8px'
          }}>
            {category}
          </h4>
          
          {NODE_TYPES
            .filter(type => type.category === category)
            .map(nodeType => (
              <div
                key={nodeType.id}
                draggable
                onDragStart={(e) => handleDragStart(e, nodeType)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '12px',
                  margin: '8px 0',
                  backgroundColor: '#f8f9fa',
                  border: '1px solid #dee2e6',
                  borderRadius: '4px',
                  cursor: 'grab',
                  userSelect: 'none'
                }}
              >
                <span style={{ fontSize: '20px', marginRight: '12px' }}>
                  {nodeType.icon}
                </span>
                <span>{nodeType.name}</span>
              </div>
            ))
          }
        </div>
      ))}
    </div>
  );
};

export default NodePalette;
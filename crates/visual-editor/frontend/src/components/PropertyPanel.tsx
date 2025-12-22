// Placeholder component for the property panel
// This will show/edit properties of selected nodes and edges

import React from 'react';

interface PropertyPanelProps {
  selectedElement?: {
    id: string;
    type: 'node' | 'edge';
    data: Record<string, any>;
  };
  onPropertyChange?: (id: string, property: string, value: any) => void;
}

export const PropertyPanel: React.FC<PropertyPanelProps> = ({
  selectedElement,
  onPropertyChange,
}) => {
  const handleInputChange = (property: string, value: any) => {
    if (selectedElement && onPropertyChange) {
      onPropertyChange(selectedElement.id, property, value);
    }
  };

  if (!selectedElement) {
    return (
      <div style={{
        width: '300px',
        height: '100%',
        backgroundColor: '#fff',
        borderLeft: '1px solid #ddd',
        padding: '16px'
      }}>
        <h3>Properties</h3>
        <p style={{ color: '#666', fontStyle: 'italic' }}>
          Select a node or edge to view its properties
        </p>
      </div>
    );
  }

  return (
    <div style={{
      width: '300px',
      height: '100%',
      backgroundColor: '#fff',
      borderLeft: '1px solid #ddd',
      padding: '16px'
    }}>
      <h3>Properties</h3>
      
      <div style={{ marginBottom: '16px' }}>
        <strong>Type:</strong> {selectedElement.type}
      </div>
      
      <div style={{ marginBottom: '16px' }}>
        <strong>ID:</strong> {selectedElement.id}
      </div>

      {/* Common properties */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
          Name:
        </label>
        <input
          type="text"
          value={selectedElement.data.name || ''}
          onChange={(e) => handleInputChange('name', e.target.value)}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px'
          }}
        />
      </div>

      {/* Node-specific properties */}
      {selectedElement.type === 'node' && (
        <>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
              Node Type:
            </label>
            <select
              value={selectedElement.data.nodeType || 'cpu'}
              onChange={(e) => handleInputChange('nodeType', e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            >
              <option value="cpu">CPU</option>
              <option value="gpu">GPU</option>
              <option value="storage">Storage</option>
              <option value="switch">Network Switch</option>
            </select>
          </div>

          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
              Cores/Units:
            </label>
            <input
              type="number"
              value={selectedElement.data.cores || 1}
              onChange={(e) => handleInputChange('cores', parseInt(e.target.value))}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
          </div>
        </>
      )}

      {/* Edge-specific properties */}
      {selectedElement.type === 'edge' && (
        <>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
              Bandwidth (Mbps):
            </label>
            <input
              type="number"
              value={selectedElement.data.bandwidth || 1000}
              onChange={(e) => handleInputChange('bandwidth', parseInt(e.target.value))}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
          </div>

          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
              Latency (ms):
            </label>
            <input
              type="number"
              step="0.1"
              value={selectedElement.data.latency || 0}
              onChange={(e) => handleInputChange('latency', parseFloat(e.target.value))}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
          </div>
        </>
      )}

      {/* Custom properties */}
      <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px solid #eee' }}>
        <h4>Custom Properties</h4>
        <p style={{ color: '#666', fontSize: '14px' }}>
          Additional properties will be displayed here based on the selected element type.
        </p>
      </div>
    </div>
  );
};

export default PropertyPanel;
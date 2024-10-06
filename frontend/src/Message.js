import React from 'react';
import { ThreeDots } from 'react-loader-spinner';
import './Message.css';

function Message({ text, type, id, question, audio, loading, onTextToSpeech }) {
  return (
    <div className={`message ${type}`}>
      <p>{text}</p>
      {type !== 'user' && (
        <div className="message-actions">
          <button onClick={() => onTextToSpeech(id, question, audio)} disabled={loading}>
            {loading ? (
              <ThreeDots color="#ffffff" height={20} width={20} />
            ) : (
              '🔊'
            )}
          </button>
        </div>
      )}
    </div>
  );
}

export default Message;

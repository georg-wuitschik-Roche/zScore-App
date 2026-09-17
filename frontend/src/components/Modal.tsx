import type { ReactNode } from 'react';

interface Props {
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** Set when something outside needs to find this specific dialog — the
   *  classes are shared, so they can't identify one. */
  id?: string;
  /** Extra class on the panel, for per-dialog sizing. */
  className?: string;
}

/** Centred dialog with a titled header, close button and click-outside dismiss.
 *  Backs both the Settings modal and the Download Plots dialog. */
export function Modal({ title, onClose, children, id, className }: Props) {
  return (
    <div className="settings-modal-backdrop" onClick={onClose}>
      <div
        className={className ? `settings-modal ${className}` : 'settings-modal'}
        id={id}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="settings-modal-header">
          <h2>{title}</h2>
          <button className="settings-modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="settings-modal-body">{children}</div>
      </div>
    </div>
  );
}

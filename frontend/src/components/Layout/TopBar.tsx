import React from "react";
import "./TopBar.css";

interface TopBarProps {
  title: string;
  breadcrumb?: string;
  showHamburger?: boolean;
  onHamburgerClick?: () => void;
}

const TopBar: React.FC<TopBarProps> = ({
  title,
  breadcrumb,
  showHamburger,
  onHamburgerClick,
}) => {
  const now = new Date().toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  });

  return (
    <header className="topbar">
      <div className="topbar__left">
        {showHamburger && (
          <button className="topbar__hamburger" onClick={onHamburgerClick}
            aria-label="Open menu">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        )}
        <h1 className="topbar__title">{title}</h1>
        {breadcrumb && (
          <span className="topbar__breadcrumb caption">{breadcrumb}</span>
        )}
      </div>
      <div className="topbar__right">
        <span className="topbar__timestamp caption">
          Last updated: {now}
        </span>
      </div>
    </header>
  );
};

export default TopBar;

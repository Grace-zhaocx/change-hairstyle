import React from 'react';
import { Layout, Typography, Button, Space } from 'antd';
import { useNavigate } from 'react-router-dom';

const { Header: AntHeader } = Layout;
const { Title } = Typography;

const Header: React.FC = () => {
  const navigate = useNavigate();

  return (
    <AntHeader className="bg-white shadow-sm px-6">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center">
          <Title
            level={3}
            className="!mb-0 text-blue-600 cursor-pointer"
            onClick={() => navigate('/')}
          >
            ✂️ AI智能换发型
          </Title>
        </div>

        {/* Navigation */}
        <Space size="middle">
          <Button type="text" onClick={() => navigate('/')}>
            首页
          </Button>
          <Button type="text" onClick={() => navigate('/upload')}>
            开始体验
          </Button>
        </Space>
      </div>
    </AntHeader>
  );
};

export default Header;
import React from 'react';
import { Layout, Typography, Row, Col, Space } from 'antd';

const { Footer: AntFooter } = Layout;
const { Text, Link } = Typography;

const Footer: React.FC = () => {
  return (
    <AntFooter className="bg-gray-900 text-gray-400 py-8">
      <div className="max-w-7xl mx-auto">
        <Row gutter={[24, 24]}>
          <Col xs={24} md={8}>
            <div className="mb-4">
              <h4 className="text-white text-lg font-semibold mb-3">
                AI智能换发型
              </h4>
              <Text className="text-sm">
                基于先进AI技术的在线换发型平台，让您轻松找到完美发型。
              </Text>
            </div>
          </Col>

          <Col xs={24} md={8}>
            <div className="mb-4">
              <h4 className="text-white text-lg font-semibold mb-3">
                快速链接
              </h4>
              <Space direction="vertical" size="small">
                <Link href="/" className="text-gray-400 hover:text-white">
                  首页
                </Link>
                <Link href="/upload" className="text-gray-400 hover:text-white">
                  开始体验
                </Link>
                <Link href="#" className="text-gray-400 hover:text-white">
                  使用教程
                </Link>
                <Link href="#" className="text-gray-400 hover:text-white">
                  常见问题
                </Link>
              </Space>
            </div>
          </Col>

          <Col xs={24} md={8}>
            <div className="mb-4">
              <h4 className="text-white text-lg font-semibold mb-3">
                关于我们
              </h4>
              <Space direction="vertical" size="small">
                <Link href="#" className="text-gray-400 hover:text-white">
                  技术介绍
                </Link>
                <Link href="#" className="text-gray-400 hover:text-white">
                  隐私政策
                </Link>
                <Link href="#" className="text-gray-400 hover:text-white">
                  服务条款
                </Link>
                <Link href="#" className="text-gray-400 hover:text-white">
                  联系我们
                </Link>
              </Space>
            </div>
          </Col>
        </Row>

        <div className="border-t border-gray-800 mt-8 pt-8 text-center">
          <Text className="text-sm">
            © 2024 AI智能换发型. 保留所有权利.
          </Text>
        </div>
      </div>
    </AntFooter>
  );
};

export default Footer;
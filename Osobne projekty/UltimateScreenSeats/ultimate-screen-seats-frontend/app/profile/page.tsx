"use client"

import { Card, CardBody, CardHeader } from "@nextui-org/card";
import { Tab, Tabs } from "@nextui-org/tabs";
import React, { Key, useState } from "react";

import AccountSettings from "@/components/profile/account-settings";
import ChangePassword from "@/components/profile/change-password";

export default function Page() {
  const [selected, setSelected] = useState("account-settings");

  const handleSelectionChange = (key: Key) => {
    setSelected(key as string);
  };

  return (
    <div className="max-w-7xl mx-auto flex flex-col justify-center mt-10 sm:px-8 px-4">
      <Card className="w-full p-8">
        <CardHeader className="p-2 flex-col items-start border-b-2 border-default-200 mb-4">
          <h1 className="text-primary text-4xl font-semibold mb-2">Profil</h1>
          <h2 className="text-default-500 text-lg">W tym miejscu możesz zarządzać swoim kontem.</h2>
        </CardHeader>
        <CardBody className="overflow-hidden flex flex-col">
          <Tabs
            aria-label="Tabs form"
            className="vertical-tabs"
            isVertical={false}
            selectedKey={selected}
            size="md"
            onSelectionChange={handleSelectionChange}
          >
            <Tab key="account-settings" title="Zaktualizuj dane">
              <AccountSettings />
            </Tab>
            <Tab key="change-password" title="Zmień hasło">
              <ChangePassword />
            </Tab>
          </Tabs>
        </CardBody>
      </Card>
    </div>
  );
}
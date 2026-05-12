from pathlib import Path
from zipfile import ZipFile
import csv
import io

import numpy as np
import cv2
from numpy.typing import NDArray


class EnsaioReader:
    def __init__(
        self,
        zip_path: Path,
        *,
        first_pulse_timestamp: int = 0,
        exposure: int = 0,
        px_p_mm: float = 0,
        pulses_period: int = 0,
    ):
        self.__zip_path = zip_path
        self.__zip = ZipFile(self.__zip_path, "r")

        calibration_filename = [
            filename
            for filename in self.__zip.namelist()
            if "calibration_data" in filename
        ][0]

        with io.TextIOWrapper(
            self.__zip.open(calibration_filename, "r"), encoding="UTF-8"
        ) as file:
            reader = csv.DictReader(file)
            data = next(reader)

            self.__first_pulse_timestamp = int(data["first_pulse_timestamp"])
            self.__exposure = int(data["exposure"])
            self.__px_p_mm = float(data["px_p_mm"])
            self.__pulses_period = int(data["pulses_period_ns"])

        self.__imu_data = []
        imu_filenames = [
            filename for filename in self.__zip.namelist() if "imu" in filename
        ]

        if len(imu_filenames) > 0:
            imu_filename = imu_filenames[0]

            with io.TextIOWrapper(
                self.__zip.open(imu_filename, "r"), encoding="UTF-8"
            ) as file:
                reader = csv.DictReader(file)

                for row in reader:
                    self.__imu_data.append(
                        {
                            "timestamp": int(row["timestamp"]),
                            "qx": float(row["qx"]) if row["qx"] else 0.0,
                            "qy": float(row["qy"]) if row["qy"] else 0.0,
                            "qz": float(row["qz"]) if row["qz"] else 0.0,
                            "qw": float(row["qw"]) if row["qw"] else 0.0,
                        }
                    )

        self.__imgs = sorted(
            [filename for filename in self.__zip.namelist() if ".jpg" in filename]
        )
        self.__img_count = len(self.__imgs)


    def get_name(self) -> str:
        return self.__zip_path.name

    def get_first_pulse_timestamp(self) -> int:
        return self.__first_pulse_timestamp

    def get_pulses_period(self) -> int:
        return self.__exposure

    def get_exposure(self) -> int:
        return self.__pulses_period

    def get_px_p_mm(self) -> float:
        return self.__px_p_mm

    def get_img_count(self) -> int:
        return self.__img_count

    def get_imu_data(self) -> [dict]:
        return self.__imu_data

    def get_img(self, i: int) -> (int, NDArray):
        filename = self.__imgs[i]
        try:
            frame = cv2.imdecode(
                    np.frombuffer(self.__zip.read(filename), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            return (int(Path(filename).stem), frame)

        except KeyboardInterrupt:
            pass

    def get_all_imgs(self) -> [(int, NDArray)]:
        imgs = []
        try:
            for filename in self.__imgs:
                frame = cv2.imdecode(
                    np.frombuffer(self.__zip.read(filename), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                imgs.append((int(Path(filename).stem), frame))
            return imgs

        except KeyboardInterrupt:
            return []
